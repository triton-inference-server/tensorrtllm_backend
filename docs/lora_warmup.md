# LoRa Warmup Example with BFloat16

This document provides an example of initializing LoRa weights and configs as warmups to the backend model so that inference can use LoRa adapters using only the `lora_task_id`.  This approach avoids the need for LoRa weights or config to be used within the requests made to the backend, and allows for bfloat16 weights to be used without needing to express them in a `python` backend model (such as `preprocessing`) where numpy conversion does not support `bfloat16`.

This example assumes that the user as pre-trained a model and has the LoRa weights and configs available from the training process as `.safetensor` files and a `config.json` file.

## Compile Base Model

The base model should be compiled according to the guidance provided in [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

## Prepare LoRa Weights as Warmup files

1. Convert to `.bin` format
   
    The expected format for lora weights by the provided conversion script, [hf_lora_convert](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/hf_lora_convert.py) assumes the existance of `adapter_config.json` and `adapter_model.bin`.  If weights are stored from training as `adapter_model.safetensors`, the following script can be used to convert the weights to the expected format.

    ```python
    import torch
    from safetensors.torch import load_file

    ADAPTER_DIR = <directory for adapter checkpoint / weights>

    torch.save(
        safetensors_load_file(
            os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
        ,
        os.path.join(ADAPTER_DIR, "adapter_model.bin),
    )
    ```

2. Prepare `config` and `weights` for TensorRT-LLM

    The [hf_lora_convert](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/hf_lora_convert.py) script can be used to convert the weights and config to the expected format for TensorRT-LLM.

    As of v0.10.0 the conversion script saves outputs in the `.npy` format only.  This can be updated by updating `write_npy=False` in the [hf_lora_convert.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/hf_lora_convert.py#L142) file.

    After allowing for the output to be saved as `.bin`

    ```python
    from hf_lora_convert import convert_hf_model

    ADAPTER_DIR = <directory for adapter checkpoint / weights>
    DTYPE = "bfloat16" # Specify adapter 

    convert_hf_model(
        folder,
        dtype="bfloat16",
        out_dir=folder
    )
    ```

    This will result in the saving of two output files `adapter_model.bin` and `adapter_config.bin`.

    These files can be used for warmup inputs to the backend model.


## Configure Warmup for `tensorrt_llm` model

After obtaining the warmup lora weights and configs from the previous steps a warmup folder should be added to the `tensorrt_llm` model directory.

1. Create warmup folder:

    Files for the warmup will be added within the model-repository which will be run using triton-inference-server 

    ```bash
    model-repository/
    ensemble/
    preprocessing/
    postprocessing/
    tensorrt_llm/
            - config.pbtxt
            - 1/
            warmup/
                - Files will be added here
    ```

1. Creating warmup files
   
   ```python
    import struct

    WARMUP_DIR = <path to warmup dir>

    # Define warmup input ids (example )
    input_ids = [123, 456, 1, 33]

    # Write to a binary file
    with open(os.path.join(WARMUP_DIR, "input_ids"), "wb") as f:
        for i in input_ids:
            f.write(struct.pack('<i', i))  # '<i' means little-endian int

    input_lengths = len(input_ids)

    # Write to a binary file
    with open(os.path.join(warmup_dir, "input_lengths"), "wb") as f:
        f.write(struct.pack('<i', input_lengths))  # '<i' means little-endian int

    # save end_id
    end_id = 128001 # Will vary based on tokenizer used

    with open(os.path.join(warmup_dir, "end_id"), "wb") as f:
        f.write(struct.pack('<i', end_id))  # '<i' means little-endian int

    # Specify output lengths (using small value to speed up warmup)
    request_output_len = 3
    with open(os.path.join(warmup_dir, "output_lengths"), "wb") as f:
        f.write(struct.pack('<i', request_output_len))  # '<i' means little-endian int

    # Specify beam width
    beam_width = 3
    with open(os.path.join(warmup_dir, "beam_width"), "wb") as f:
        f.write(struct.pack('<i', beam_width))  # '<i' means little-endian int

    # Specify lora_task_id(s)
    n_adapters = 3
    for lora_task_id in range(n_adapters):
        # Write to a binary file
        with open(os.path.join(warmup_dir, f"lora_id_{lora_task_id}"), "wb") as f:
            f.write(struct.pack('<q', lora_task_id))
    ```

    The above script will create the necessary files for warmup.  The `input_ids` should be updated to reflect the input_ids that will be used for warmup.  The `end_id` should be updated to reflect the end_id used by the tokenizer.  The `request_output_len` and `beam_width` should be set to the desired values for warmup and match the complation parameters which were performed on the base model.  The `n_adapters` should be set to the number of adapters that will be used for warmup.

    The converted `adapter_model.bin` and `adapter_config.bin` should be copied to the warmup directory but renamed for each adapter being used.  For this example we will assume that there are 3 adapters and the files are renamed resulting in the following contents of the `warmup` directory:

    ```bash
    warmup/
        - input_ids
        - input_lengths
        - end_id
        - output_lengths
        - beam_width
        - lora_id_0
        - lora_id_1
        - lora_id_2
        - adapter_model_0.bin
        - adapter_config_0.bin
        - adapter_model_1.bin
        - adapter_config_1.bin
        - adapter_model_2.bin
        - adapter_config_2.bin

    ```

 1. Updating the model `config.pbtxt`

    The configuration file for the `tensorrt_llm` model can then be updated to perform the warmup.  The `config.pbtxt` file should be updated to include the warmup configuration.  
    
    The dimensions of the adapter must be known to provide shapes within the configuration. This can be inspected by reading the `adapter_model.bin` file.

    The following is an example of the `config.pbtxt` file with the warmup configuration added:

    ```pbtxt

    model_warmup [
    {
        name: "lora_0_warmup"
        batch_size: 1
        inputs: {
        key: "lora_task_id"
        value: {
            data_type: TYPE_UINT64
            dims: [ 1 ]
            input_data_file: "lora_id_0"
        }
        }
        inputs: {
        key: "lora_weights"
        value: {
            data_type: TYPE_BF16 # This should match the datatype of the adapter
            dims: [ 224,  589824] # This should match the dimensions of the adapter
            input_data_file: "adapter_model_0.bin"
        }
        }
        inputs: {
        key: "end_id"
        value: {
            data_type: TYPE_UINT32
            dims: [ 1 ]
            input_data_file: "end_id"
        }
        }
        inputs: {
        key: "lora_config"
        value: {
            data_type: TYPE_INT32
            dims: [ 224, 3 ] # This should match the dimensions of the adapter
            input_data_file: "adapter_config_0.bin"
        }
        }
        inputs: {
        key: "input_ids"
        value: {
            data_type: TYPE_INT32
            dims: [ 4 ]
            input_data_file: "input_ids"
        }
        }
        inputs: {
        key: "input_lengths"
        value: {
            data_type: TYPE_INT32
            dims: [ 1 ]
            input_data_file: "input_lengths"
        }
        }
        inputs: {
        key: "request_output_len"
        value: {
            data_type: TYPE_UINT32
            dims: [ 1 ]
            input_data_file: "output_lengths"
        }
        }
        inputs: {
        key: "beam_width"
        value: {
            data_type: TYPE_UINT32
            dims: [ 1 ]
            input_data_file: "beam_width"
        }
        }
    },
    ... # repeat for other two adapters
    ]

    ```

1. Startup and calling

    After the model has been warmed up using this process calls can be made within the normal triton-inference-server environment while only requiring passing `lora_task_id` to the model.  The model will use the lora weights associated with the `lora_task_id` to perform inference as defined from the warmups.