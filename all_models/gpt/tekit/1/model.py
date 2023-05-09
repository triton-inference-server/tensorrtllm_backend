import json
import os

import torch
import triton_python_backend_utils as pb_utils
from torch import from_numpy

import tekit


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model assuming
        the server was not started with `--disable-auto-complete-config`. Implementing
        this function is optional. No implementation of `auto_complete_config` will
        do nothing. This function can be used to set `max_batch_size`, `input` and
        `output` properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with minimal
        model configuration in absence of a configuration file. This function returns
        the `pb_utils.ModelConfig` object with these properties. You can use the `as_dict`
        function to gain read-only access to the `pb_utils.ModelConfig` object.
        The `pb_utils.ModelConfig` object being returned from here will be used as
        the final configuration for the model.

        Note: The Python interpreter used to invoke this function will be destroyed
        upon returning from this function and as a result none of the objects created
        here will be available in the `initialize`, `execute`, or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build upon
          the configuration given by this object when setting the properties for
          this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        return auto_complete_model_config

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
        self.max_out_len = int(
            model_config['parameters']['max_out_len']['string_value'])
        config_path = os.path.join(engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.use_gpt_attention_plugin = config['plugin_config'][
            'gpt_attention_plugin']
        dtype = 'float16' if config['builder_config']['fp16'] else 'float32'
        world_size = config['builder_config']['tensor_parallel']
        assert world_size == tekit.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tekit.mpi_world_size()})'
        self.num_heads = config['builder_config']['num_heads'] // world_size
        self.hidden_size = config['builder_config']['hidden_size'] // world_size
        self.vocab_size = config['builder_config']['vocab_size']
        self.num_layers = config['builder_config']['num_layers']

        self.comm = tekit.mpi_comm()
        self.rank = tekit.mpi_rank()
        runtime_mapping = tekit.Mapping(world_size, self.rank)
        torch.cuda.set_device(self.rank % runtime_mapping.gpus_per_node)

        engine_name = tekit.rank_engine_file('gpt', dtype, world_size,
                                             self.rank)
        serialize_path = os.path.join(engine_dir, engine_name)

        tekit.init()
        runtime = tekit.GPTRuntime(dtype)
        runtime.prepare(runtime_mapping, serialize_path)
        self.decoder = tekit.GPTDecoder(runtime_mapping, runtime)

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
        input_name = 'input_ids'
        for request in requests:
            # Perform inference on the request and append it to responses list...
            if self.rank == 0:
                # Triton tensor -> numpy tensor -> PyTorch tensor
                input_ids = from_numpy(
                    pb_utils.get_input_tensor_by_name(request,
                                                      input_name).as_numpy())

            else:
                input_ids = None

            # Broadcast requests to other clients
            input_ids = self.comm.bcast(input_ids, root=0).cuda()
            self.decoder.setup(input_ids.size(0),
                               input_ids.size(1),
                               self.max_out_len,
                               self.vocab_size,
                               self.num_layers,
                               self.num_heads,
                               self.hidden_size,
                               use_plugin=self.use_gpt_attention_plugin)
            output_ids = self.decoder.decode(input_ids)

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
