import json
import os

import torch
import triton_python_backend_utils as pb_utils
from torch import from_numpy

import tensorrt_llm
from tensorrt_llm.runtime import Session
from tensorrt_llm.runtime import TensorInfo
from tensorrt_llm.functional import str_dtype_to_trt
import tensorrt as trt

def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)

def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return mpi_comm().Get_rank()


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def get_input_tensor_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> PyTorch tensor
        return from_numpy(tensor.as_numpy())
    else:
        return tensor


def get_input_scalar_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> first scalar
        tensor = tensor.as_numpy()
        return tensor.reshape((tensor.size, ))[0]
    else:
        return tensor


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.logger.log_info("Info Msg!")

        model_config = json.loads(args['model_config'])
        engine_dir = model_config['parameters']['engine_dir']['string_value']

        config_path = os.path.join(engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        dtype = config['builder_config']['precision']
        world_size = config['builder_config']['tensor_parallel']
        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

        model_name = config['builder_config']['name']
        runtime_rank = tensorrt_llm.mpi_rank() if world_size > 1 else 0

        runtime_mapping = tensorrt_llm.Mapping(world_size,
                                            runtime_rank,
                                            tp_size=world_size)
        serialize_path = get_engine_name(model_name, dtype, world_size,
                                        runtime_rank)
        serialize_path = os.path.join(engine_dir, serialize_path)

        self.stream = torch.cuda.current_stream().cuda_stream
        print(f'Loading engine from {serialize_path}')
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()
        print(f'Creating session from engine')
        self.session = Session.from_serialized_engine(engine_buffer)

        self.comm = mpi_comm()
        self.rank = mpi_rank()
        torch.cuda.set_device(self.rank % runtime_mapping.gpus_per_node)

        if self.rank != 0:
            while (True):
                self.execute([None])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            inputs = {}
            if self.rank == 0:
                inputs['input_ids'] = get_input_tensor_by_name(
                    request, 'input_ids')
                inputs['input_lengths'] = get_input_tensor_by_name(
                    request, 'input_lengths')

            # Broadcast requests to other clients
            inputs = self.comm.bcast(inputs, root=0)

            input_ids = inputs['input_ids'].cuda()

            batch_dim = input_ids.shape[0]
            input_len = input_ids.shape[1]
            input_lengths = inputs['input_lengths'].cuda()

            inputs = {
                'input_ids': input_ids,
                'input_lengths': input_lengths,
                # 'token_type_ids': token_type_ids
            }

            # print(f'input_ids.size(): {input_ids.size()}')
            output_info = self.session.infer_shapes([
                TensorInfo('input_ids', str_dtype_to_trt('int32'),
                           (batch_dim, input_len)),
                TensorInfo('input_lengths', str_dtype_to_trt('int32'), (batch_dim, ))
            ])
            # self.session._print_engine_info()

            outputs = {
                t.name: torch.empty(tuple(t.shape),
                                    dtype=trt_dtype_to_torch(t.dtype),
                                    device='cuda')
                for t in output_info
            }
            output_name = 'logits'
            assert output_name in outputs, f'{output_name} not found in outputs, check if build.py set the name correctly'
            ok = self.session.run(inputs, outputs, self.stream)

            assert ok, "Runtime execution failed"

            logits = outputs[output_name]
            logits = logits.to(dtype=torch.float32)

            if self.rank == 0:
                # Create output tensors. You need pb_utils.Tensor
                # objects to create pb_utils.InferenceResponse.
                torch.cuda.synchronize()
                self.logger.log(f'logits: {logits.cpu().numpy()}', self.logger.INFO)
                logits = [
                    pb_utils.Tensor("logits",
                                    logits.cpu().numpy())
                ]

                # Create InferenceResponse. You can set an error here in case
                # there was a problem with handling this inference request.
                # Below is an example of how you can set errors in inference
                # response:
                #
                # pb_utils.InferenceResponse(
                #    output_tensors=..., TritonError("An error occurred"))

                inference_response = pb_utils.InferenceResponse(logits)
            else:
                inference_response = pb_utils.InferenceResponse([])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        return
