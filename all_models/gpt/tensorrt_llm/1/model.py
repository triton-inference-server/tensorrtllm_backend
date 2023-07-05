import json
import os

import tensorrt_llm
import torch
import triton_python_backend_utils as pb_utils
from tensorrt_llm.runtime import GenerationSession, ModelConfig, SamplingConfig
from torch import from_numpy


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
        model_config = json.loads(args['model_config'])
        engine_dir = model_config['parameters']['engine_dir']['string_value']
        config_path = os.path.join(engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        use_gpt_attention_plugin = config['plugin_config'][
            'gpt_attention_plugin']
        self.remove_input_padding = config['plugin_config'][
            'remove_input_padding']
        model = config['builder_config']['name']
        dtype = config['builder_config']['precision']
        world_size = config['builder_config']['tensor_parallel']
        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
        num_heads = config['builder_config']['num_heads'] // world_size
        hidden_size = config['builder_config']['hidden_size'] // world_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        multi_query_mode = False
        if 'multi_query_mode' in config['builder_config'].keys():
            multi_query_mode = config['builder_config']['multi_query_mode']

        self.comm = mpi_comm()
        self.rank = mpi_rank()

        model_config = ModelConfig(
            num_heads=num_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            multi_query_mode=multi_query_mode,
            remove_input_padding=self.remove_input_padding)
        engine_name = get_engine_name(model, dtype, world_size, self.rank)
        serialize_path = os.path.join(engine_dir, engine_name)
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()
        runtime_mapping = tensorrt_llm.Mapping(world_size, self.rank)
        torch.cuda.set_device(self.rank % runtime_mapping.gpus_per_node)
        self.decoder = GenerationSession(model_config, engine_buffer,
                                         runtime_mapping)

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
                inputs['request_output_len'] = get_input_scalar_by_name(
                    request, 'request_output_len')
                inputs['beam_width'] = get_input_scalar_by_name(
                    request, 'beam_width')
                inputs['temperature'] = get_input_scalar_by_name(
                    request, 'temperature')
                inputs['runtime_top_k'] = get_input_scalar_by_name(
                    request, 'runtime_top_k')
                inputs['runtime_top_p'] = get_input_scalar_by_name(
                    request, 'runtime_top_p')
                inputs['len_penalty'] = get_input_scalar_by_name(
                    request, 'len_penalty')
                inputs['repetition_penalty'] = get_input_scalar_by_name(
                    request, 'repetition_penalty')
                inputs[
                    'beam_search_diversity_rate'] = get_input_scalar_by_name(
                        request, 'beam_search_diversity_rate')
                inputs['random_seed'] = get_input_scalar_by_name(
                    request, 'random_seed')
                inputs['top_p_decay'] = get_input_scalar_by_name(
                    request, 'top_p_decay')
                inputs['top_p_min'] = get_input_scalar_by_name(
                    request, 'top_p_min')
                inputs['top_p_reset_ids'] = get_input_scalar_by_name(
                    request, 'top_p_reset_ids')

            # Broadcast requests to other clients
            inputs = self.comm.bcast(inputs, root=0)
            input_ids = inputs['input_ids'].cuda()
            input_lengths = inputs['input_lengths'].cuda()
            sampling_config = SamplingConfig(
                end_id=50256,
                pad_id=50256,
                num_beams=inputs['beam_width'],
                temperature=inputs['temperature'],
                top_k=inputs['runtime_top_k'],
                top_p=inputs['runtime_top_p'],
                length_penalty=inputs['len_penalty'],
                repetition_penalty=inputs['repetition_penalty'],
            )
            if self.remove_input_padding:
                self.decoder.setup(
                    batch_size=1,
                    max_input_length=torch.max(input_lengths).item(),
                    max_new_tokens=inputs['request_output_len'])
            else:
                self.decoder.setup(batch_size=input_ids.size(0),
                                   max_input_length=input_ids.size(1),
                                   max_new_tokens=inputs['request_output_len'])
            output_ids = self.decoder.decode(input_ids, input_lengths,
                                             sampling_config)

            if self.rank == 0:
                # Create output tensors. You need pb_utils.Tensor
                # objects to create pb_utils.InferenceResponse.
                torch.cuda.synchronize()
                output_ids = pb_utils.Tensor("output_ids",
                                             output_ids.cpu().numpy())

                # Create InferenceResponse. You can set an error here in case
                # there was a problem with handling this inference request.
                # Below is an example of how you can set errors in inference
                # response:
                #
                # pb_utils.InferenceResponse(
                #    output_tensors=..., TritonError("An error occured"))

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_ids])
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
