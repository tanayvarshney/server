<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Triton Inference Server Documentation

| [Installation](README.md#installation) | [Getting Started](README.md#getting-started) | [User Guide](README.md#user-guide) | [API Guide](doc_files/protocol/README.md) | [Additional Resources](README.md#resources) | [Customization Guide](README.md#customization-guide) |
| ------------ | --------------- | --------------- | ------------ | --------------- | --------------- | 

## Installation
Before you can use the Triton Docker image you must install
[Docker](https://docs.docker.com/engine/install). If you plan on using
a GPU for inference you must also install the [NVIDIA Container
Toolkit](https://github.com/NVIDIA/nvidia-docker). DGX users should
follow [Preparing to use NVIDIA
Containers](http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html).

Pull the image using the following command.

```
$ docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3
```

Where \<yy.mm\> is the version of Triton that you want to pull. For a complete list of all the variants and versions of the Triton Inference Server Container,  visit the [NGC Page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). More information about customizing the Triton Container can be found in [this section](https://github.com/triton-inference-server/server/blob/main/docs/compose.md) of the User Guide.

## Getting Started

This guide covers the simplest possible workflow for deploying a model using a Triton Inference Server.

- [Create Model Repository](doc_files/quickstart.md#create-a-model-repository)
- [Run Triton](doc_files/quickstart.md#run-triton)
- [Run a sample Client](doc_files/quickstart.md#running-a-sample-client)

Triton Inference Server has a considerble list versrtile and powerful features. All new users are recommended to explore the [User Guide](README.md#user-guide) and the [additional resources](README.md#resources) sections for features most relevant to their usecase. 

## User Guide
The User Guide describes how to use Triton as an inference solution, including information on how to configure Triton, how to organize and configure your models, how to use the C++ and Python clients, etc. 

- **[Model Repository](doc_files/model_repository.md)**: Model Repositories are the organizational hub for using Triton. All models, configuration files, and additional resources need specifically to serve the models are housed inside a model repository.
  - [Cloud Storage](doc_files/model_repository.md#model-repository-locations)
  - [File Organization](doc_files/model_repository.md#model-files)
  - [Model Versioning](doc_files/model_repository.md#model-versions)
- **[Model Configuration](doc_files/model_configuration.md)**: A Model Configuration file is the primary point of contact for all the model level tweaks, whether it is about reshaping the output tensor or directing Triton to build dynamic batch, all model level "knobs" are handled in a configuration file. 
  - [Required Model Configuration](doc_files/model_configuration.md#minimal-model-configuration)
    - [Maximum Batch Size - Batching and Non-Batching Models](doc_files/model_configuration.md#maximum-batch-size)
    - [Input and Output Tensors](doc_files/model_configuration.md#inputs-and-outputs)
      - [Tensor Datatypes](doc_files/model_configuration.md#datatypes)
      - [Tensor Reshape](doc_files/model_configuration.md#reshape)
      - [Shape Tensor](doc_files/model_configuration.md#shape-tensors)
  - [Auto-Generate Required Model Configuration](doc_files/model_configuration.md#auto-generated-model-configuration)
  - [Version Policy](doc_files/model_configuration.md#version-policy)
  - [Instance Groups](doc_files/model_configuration.md#instance-groups)
    - [Specifying Multiple Model Instances](doc_files/model_configuration.md#multiple-model-instances)
    - [CPU and GPU Instances](doc_files/model_configuration.md#cpu-model-instance)
    - [Configuring Rate Limiter](doc_files/model_configuration.md#rate-limiter-configuration)
  - [Optimization Settings](doc_files/model_configuration.md#optimization_policy)
    - [Framework-Specific Optimization](doc_files/optimization.md#framework-specific-optimization)
      - [ONNX-TensorRT](doc_files/optimization.md#onnx-with-tensorrt-optimization-ort-trt)
      - [ONNX-OpenVINO](doc_files/optimization.md#onnx-with-openvino-optimization)
      - [TensorFlow-TensorRT](doc_files/optimization.md#tensorflow-with-tensorrt-optimization-tf-trt)
      - [TensorFlow-Mixed-Precision](doc_files/optimization.md#tensorflow-automatic-fp16-optimization)
    - [NUMA Optimization](doc_files/optimization.md#numa-optimization)
  - [Scheduling and Batching](doc_files/model_configuration.md#scheduling-and-batching)
    - [Default Scheduler - Non-Batching](doc_files/model_configuration.md#default-scheduler)
    - [Dynamic Batcher](doc_files/model_configuration.md#dynamic-batcher)
      - [How to Configure Dynamic Batcher](doc_files/model_configuration.md#recommended-configuration-process)
        - [Delayed Batching](doc_files/model_configuration.md#delayed-batching)
        - [Preferred Batch Size](doc_files/model_configuration.md#preferred-batch-sizes)
      - [Preserving Request Ordering](doc_files/model_configuration.md#preserve-ordering)
      - [Priority Levels](doc_files/model_configuration.md#priority-levels)
      - [Queuing Policies](doc_files/model_configuration.md#queue-policy)
      - [Ragged Batching](doc_files/ragged_batching.md)
    - [Sequence Batcher](doc_files/model_configuration.md#sequence-batcher)
      - [Stateful Models](doc_files/architecture.md#stateful-models)
      - [Control Inputs](doc_files/architecture.md#control-inputs)
      - [Implicit State - Stateful Inference Using a Stateless Model](doc_files/architecture.md#implicit-state-management)
      - [Sequence Scheduling Strategies](doc_files/architecture.md#scheduling-strateties)
        - [Direct](doc_files/architecture.md#direct)
        - [Oldest](doc_files/architecture.md#oldest)
    - [Rate Limiter](doc_files/rate_limiter.md)
  - [Model Warmup](doc_files/model_configuration.md#model-warmup)
  - [Inference Request/Response Cache](doc_files/model_configuration.md#response-cache)
- **Model Pipeline**: Building ensembles is as easy as adding an addition configuration file which outlines the specific flow of tensors from one model to another. Further changes in existing (individual)model configurations might be needed base on the use case. 
  - [Model Ensemble](doc_files/architecture.md#ensemble-models)
  - [Business Logic Scripting (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- **[Model Management](doc_files/model_management.md)**: Users can specify policies in the model configuration for loading and unloading of models. This section covers user selectable policy details.  
  - [Explicit Model Loading and Unloading](doc_files/model_management.md#model-control-mode-explicit)
  - [Modifying the Model Repository](doc_files/model_management.md#modifying-the-model-repository)
- **[Metrics](doc_files/metrics.md)** Triton provides Prometheus metrics. Learn about availble metrics. 
- **[Framework Custom Operations](doc_files/custom_operations.md)**: Some frameworks provide the option of building custom layers/operations. These can be added to specific Triton Backends for the those frameworks.
  - [TensorRT](doc_files/custom_operations.md#tensorrt)
  - [TensorFlow](doc_files/custom_operations.md#tensorflow)
  - [PyTorch](doc_files/custom_operations.md#pytorch)
  - [ONNX](doc_files/custom_operations.md#onnx)
- **[Client Libraries and Examples](https://github.com/triton-inference-server/client)** Use the Triton Client API to integrate client applications over the network HTTP/gRPC API or integrate applications directly with Triton using CUDA shared memory to remove network overhead.
  - [C++ HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
  - [Python HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
  - [Java HTTP Library](https://github.com/triton-inference-server/client/tree/main/src/java)
  - GRPC Generated Libraries
    - [go](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/go)
    - [Java/Scala](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/java)
    - [Javascript](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/javascript)
  - [Shared Memory Extention](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md)
- **[Performance Analysis](doc_files/optimization.md)**: Understanding Inference perfomance is key to better resource utilization. Use Triton's Tools to costomize your deployment.
  - [Model Analyzer](doc_files/model_analyzer.md)
  - [Performance Analyzer](doc_files/perf_analyzer.md)
  - [Inference Request Tracing](doc_files/trace.md)
- **[Jetson and JetPack](doc_files/jetson.md)** Triton can be deployed on edge devices. Explore resources and examples.

## Resources

The following resources are recommended to explore the full suite of Triton Inference Server's functionalities.
- **Clients**: Triton Inference Server comes with C++, Python and Java APIs with which users can send HTTP/REST or gRPC(possible extensions for other languages) requests. Explore the [client repository](https://github.com/triton-inference-server/server/tree/main/docs/protocol) for examples and documentation.

- **Configuring Deployment**: Triton comes with three tools which can be used to configure deployment setting, measure performance and recommend optimizations.
  - [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) Model Analyzer is CLI tool built to recommend deployment configurations for Triton Inference Server based on user's Quality of Service Requirements. It also generates detailed reports about model performance to summarize the benefits and trade offs of different configurations.
  - [Perf Analyzer](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md): Perf Analyzer is a CLI application built to generate inference requests and measures the latency of those requests and throughput of the model being served .
  - [Model Navigator](https://github.com/triton-inference-server/model_navigator)

- **Backends**: Triton has suports a wide varity of frameworks used to run models. Users can extend this functionality by creating custom backends.
  - [PyTorch](https://github.com/triton-inference-server/pytorch_backend): Widely used Open Source DL Framework
  - [TensorFlow](https://github.com/triton-inference-server/tensorflow_backend): Widely used Open Source DL Framework
  - [TensorRT](https://github.com/triton-inference-server/tensorrt_backend): NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) is an inference acceleration SDK that provide a with range of graph optimizations, kernel optimization, use of lower precision, and more.
  - [ONNX](https://github.com/triton-inference-server/onnxruntime_backend): ONNX Runtime is a cross-platform inference and training machine-learning accelerator.
  - [OpenVINO](https://github.com/triton-inference-server/openvino_backend): OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.
  - [Paddle Paddle](https://github.com/triton-inference-server/paddlepaddle_backend): Widely used Open Source DL Framework
  - [Python](https://github.com/triton-inference-server/python_backend): Users can add custom business logic, or any python code/model for serving requests.
  - [Forest Inference Library](https://github.com/triton-inference-server/fil_backend): Backend built for forest models trained by several popular machine learning frameworks (including XGBoost, LightGBM, Scikit-Learn, and cuML)
  - [DALI](https://github.com/triton-inference-server/dali_backend): NVIDIA [DALI](https://developer.nvidia.com/dali) is a Data Loading Library purpose built to accelerated pre-processing and data loading steps in a Deep Learning Pipeline.
  - [HugeCTR](https://github.com/triton-inference-server/hugectr_backend): HugeCTR is a GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates
  - [Managed Stateful Models](https://github.com/triton-inference-server/stateful_backend): This backend automatically manages the input and output states of a model. The states are associated with a sequence id and need to be tracked for inference requests associated with the sequence id.
  - [Faster Transformer](https://github.com/triton-inference-server/fastertransformer_backend): NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/) (FT) is a library implementing an accelerated engine for the inference of transformer-based neural networks, with a special emphasis on large models, spanning many GPUs and nodes in a distributed manner.
  - [Building Custom Backends](https://github.com/triton-inference-server/backend/tree/main/examples#tutorial)
  - [Sample Custom Backend: Repeat_backend](https://github.com/triton-inference-server/repeat_backend): Backend built to demonstrate sending of zero, one, or multiple responses per request.



## Customization Guide
This guide describes how to build and test Triton and also how Triton can be extended with new functionality.

- [Build](build.md)
- [Protocols and APIs](inference_protocols.md).
- [Backends](https://github.com/triton-inference-server/backend)
- [Repository Agents](repository_agents.md)
- [Test](test.md)