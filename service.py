from __future__ import annotations

import os, subprocess, typing, asyncio, sys, logging, bentoml, httpx, fastapi, pydantic

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, PlainTextResponse


logger = logging.getLogger('bentoml.service')


class BentoArgs(pydantic.BaseModel):
  model_repository: str = pydantic.Field(
    default=os.path.join(os.path.dirname(__file__), 'model_repository'),
    description='Path to model repository directory. It may be specified multiple times to add multiple model repositories. Note that if a model is not unique across all model repositories at any time, the model will not be available.',
  )
  model_control_mode: typing.Literal['none', 'poll', 'explicit'] = pydantic.Field(
    default='none',
    description='Specify the mode for model management. Options are "none", "poll" and "explicit". The default is "none". For "none", the server will load all models in the model repository(s) at startup and will not make any changes to the load models after that. For "poll", the server will poll the model repository(s) to detect changes and will load/unload models based on those changes. The poll rate is controlled by \'repository-poll-secs\'. For "explicit", model load and unload is initiated by using the model control APIs, and only models specified with --load-model will be loaded at startup.',
  )
  load_model: list[str] | None = pydantic.Field(
    default=None,
    description="Name of the model to be loaded on server startup. It may be specified multiple times to add multiple models. To load ALL models at startup, specify '*' as the model name with --load-model=* as the ONLY --load-model argument, this does not imply any pattern matching. Specifying --load-model=* in conjunction with another --load-model argument will result in error. Note that this option will only take effect if --model-control-mode=explicit is true.",
  )
  exit_on_error: bool = pydantic.Field(
    default=True, description='Exit the inference server if an error occurs during initialization.'
  )
  allow_http: bool = pydantic.Field(default=True, description='Allow the server to listen for HTTP requests.')
  http_address: str = pydantic.Field(
    default='0.0.0.0', description='The address for the http server to bind to. Default is 0.0.0.0'
  )
  http_port: int = pydantic.Field(
    default=8000, description='The port for the server to listen on for HTTP requests. Default is 8000.'
  )
  reuse_http_port: bool = pydantic.Field(
    default=False,
    description='Allow multiple servers to listen on the same HTTP port when every server has this option set. If you plan to use this option as a way to load balance between different Triton servers, the same model repository or set of models must be used for every server.',
  )
  allow_grpc: bool = pydantic.Field(default=True, description='Allow the server to listen for GRPC requests.')
  grpc_address: str = pydantic.Field(
    default='0.0.0.0', description='The address for the grpc server to binds to. Default is 0.0.0.0'
  )
  grpc_port: int = pydantic.Field(
    default=8001, description='The port for the server to listen on for GRPC requests. Default is 8001.'
  )
  reuse_grpc_port: bool = pydantic.Field(
    default=True,
    description='Allow multiple servers to listen on the same GRPC port when every server has this option set. If you plan to use this option as a way to load balance between different Triton servers, the same model repository or set of models must be used for every server.',
  )
  allow_metrics: bool = pydantic.Field(default=True, description='Allow the server to provide prometheus metrics.')
  allow_gpu_metrics: bool = pydantic.Field(
    default=True, description='Allow the server to provide GPU metrics. Ignored unless --allow-metrics is true.'
  )
  allow_cpu_metrics: bool = pydantic.Field(
    default=True, description='Allow the server to provide CPU metrics. Ignored unless --allow-metrics is true.'
  )
  metrics_port: int = pydantic.Field(
    default=8002, description='The port reporting prometheus metrics. Default is 8002.'
  )
  exit_timeout_secs: int = pydantic.Field(
    default=5,
    description='Timeout (in seconds) when exiting to wait for in-flight inferences to finish. After the timeout expires the server exits even if inferences are still in flight.',
  )

  id: str | None = pydantic.Field(default=None, description='Identifier for this server.')
  log_verbose: int | None = pydantic.Field(
    default=None,
    description='Set verbose logging level. Zero (0) disables verbose logging and values >= 1 enable verbose logging.',
  )
  log_info: bool | None = pydantic.Field(default=None, description='Enable/disable info-level logging.')
  log_warning: bool | None = pydantic.Field(default=None, description='Enable/disable warning-level logging.')
  log_error: bool | None = pydantic.Field(default=None, description='Enable/disable error-level logging.')
  log_format: str | None = pydantic.Field(
    default=None,
    description='Set the logging format. Options are "default" and "ISO8601". The default is "default". For "default", the log severity (L) and timestamp will be logged as "LMMDD hh:mm:ss.ssssss". For "ISO8601", the log format will be "YYYY-MM-DDThh:mm:ssZ L".',
  )
  log_file: str | None = pydantic.Field(
    default=None,
    description='Set the name of the log output file. If specified, log outputs will be saved to this file. If not specified, log outputs will stream to the console.',
  )
  disable_auto_complete_config: bool | None = pydantic.Field(
    default=None,
    description='If set, disables the triton and backends from auto completing model configuration files. Model configuration files must be provided and all required configuration settings must be specified.',
  )
  strict_readiness: bool | None = pydantic.Field(
    default=None,
    description='If true /v2/health/ready endpoint indicates ready if the server is responsive and all models are available. If false /v2/health/ready endpoint indicates ready if server is responsive even if some/all models are unavailable.',
  )
  repository_poll_secs: int | None = pydantic.Field(
    default=None,
    description='Interval in seconds between each poll of the model repository to check for changes. Valid only when --model-control-mode=poll is specified.',
  )
  model_config_name: str | None = pydantic.Field(
    default=None,
    description='The custom configuration name for models to load.The name should not contain any space character.For example: --model-config-name=h100. If --model-config-name is not set, Triton will use the default config.pbtxt.',
  )
  model_load_thread_count: int | None = pydantic.Field(
    default=None,
    description='The number of threads used to concurrently load models in model repositories. Default is 4.',
  )
  model_load_retry_count: int | None = pydantic.Field(
    default=None, description='The number of retry to load a model in model repositories. Default is 0.'
  )
  model_namespacing: bool | None = pydantic.Field(
    default=None,
    description='Whether model namespacing is enable or not. If true, models with the same name can be served if they are in different namespace.',
  )
  enable_peer_access: bool | None = pydantic.Field(
    default=None,
    description="Whether the server tries to enable peer access or not. Even when this options is set to true,  peer access could still be not enabled because the underlying system doesn't support it. The server will log a warning in this case. Default is true.",
  )
  http_header_forward_pattern: str | None = pydantic.Field(
    default=None,
    description='The regular expression pattern that will be used for forwarding HTTP headers as inference request parameters.',
  )
  http_thread_count: int | None = pydantic.Field(default=None, description='Number of threads handling HTTP requests.')
  http_restricted_api: str | None = pydantic.Field(
    default=None,
    description='Specify restricted HTTP api setting. The format of this flag is --http-restricted-api=<apis>,<key>=<value>. Where <api> is a comma-separated list of apis to be restricted. <key> will be additional header key to be checked when a HTTP request is received, and <value> is the value expected to be matched. Allowed APIs: health, metadata, inference, shared-memory, model-config, model-repository, statistics, trace, logging',
  )
  grpc_header_forward_pattern: str | None = pydantic.Field(
    default=None,
    description='The regular expression pattern that will be used for forwarding GRPC headers as inference request parameters.',
  )
  grpc_infer_allocation_pool_size: int | None = pydantic.Field(
    default=None,
    description="The maximum number of states (inference request/response queues) that remain allocated for reuse. As long as the number of in-flight requests doesn't exceed this value there will be no allocation/deallocation of request/response objects.",
  )
  grpc_max_response_pool_size: int | None = pydantic.Field(
    default=None,
    description='The maximum number of inference response objects that can remain allocated in the response queue at any given time.',
  )
  grpc_use_ssl: bool | None = pydantic.Field(
    default=None, description='Use SSL authentication for GRPC requests. Default is false.'
  )
  grpc_use_ssl_mutual: bool | None = pydantic.Field(
    default=None,
    description="Use mututal SSL authentication for GRPC requests. This option will preempt '--grpc-use-ssl' if it is also specified. Default is false.",
  )
  grpc_server_cert: str | None = pydantic.Field(
    default=None, description='File holding PEM-encoded server certificate. Ignored unless --grpc-use-ssl is true.'
  )
  grpc_server_key: str | None = pydantic.Field(
    default=None, description='File holding PEM-encoded server key. Ignored unless --grpc-use-ssl is true.'
  )
  grpc_root_cert: str | None = pydantic.Field(
    default=None, description='File holding PEM-encoded root certificate. Ignore unless --grpc-use-ssl is false.'
  )
  grpc_infer_response_compression_level: str | None = pydantic.Field(
    default=None,
    description='The compression level to be used while returning the infer response to the peer. Allowed values are none, low, medium and high. By default, compression level is selected as none.',
  )
  grpc_keepalive_time: int | None = pydantic.Field(
    default=None,
    description='The period (in milliseconds) after which a keepalive ping is sent on the transport. Default is 7200000 (2 hours).',
  )
  grpc_keepalive_timeout: int | None = pydantic.Field(
    default=None,
    description='The period (in milliseconds) the sender of the keepalive ping waits for an acknowledgement. If it does not receive an acknowledgment within this time, it will close the connection. Default is 20000 (20 seconds).',
  )
  grpc_keepalive_permit_without_calls: bool | None = pydantic.Field(
    default=None,
    description='Allows keepalive pings to be sent even if there are no calls in flight (0 : false; 1 : true). Default is 0 (false).',
  )
  grpc_http2_max_pings_without_data: int | None = pydantic.Field(
    default=None,
    description='The maximum number of pings that can be sent when there is no data/header frame to be sent. gRPC Core will not continue sending pings if we run over the limit. Setting it to 0 allows sending pings without such a restriction. Default is 2.',
  )
  grpc_http2_min_recv_ping_interval_without_data: int | None = pydantic.Field(
    default=None,
    description="If there are no data/header frames being sent on the transport, this channel argument on the server side controls the minimum time (in milliseconds) that gRPC Core would expect between receiving successive pings. If the time between successive pings is less than this time, then the ping will be considered a bad ping from the peer. Such a ping counts as a 'ping strike'. Default is 300000 (5 minutes).",
  )
  grpc_http2_max_ping_strikes: int | None = pydantic.Field(
    default=None,
    description='Maximum number of bad pings that the server will tolerate before sending an HTTP2 GOAWAY frame and closing the transport. Setting it to 0 allows the server to accept any number of bad pings. Default is 2.',
  )
  grpc_max_connection_age: int | None = pydantic.Field(
    default=None, description='Maximum time that a channel may exist in milliseconds.Default is undefined.'
  )
  grpc_max_connection_age_grace: int | None = pydantic.Field(
    default=None, description='Grace period after the channel reaches its max age. Default is undefined.'
  )
  grpc_restricted_protocol: str | None = pydantic.Field(
    default=None,
    description='Specify restricted GRPC protocol setting. The format of this flag is --grpc-restricted-protocol=<protocols>,<key>=<value>. Where <protocol> is a comma-separated list of protocols to be restricted. <key> will be additional header key to be checked when a GRPC request is received, and <value> is the value expected to be matched. Allowed protocols: health, metadata, inference, shared-memory, model-config, model-repository, statistics, trace, logging',
  )
  allow_sagemaker: bool | None = pydantic.Field(
    default=None, description='Allow the server to listen for Sagemaker requests. Default is false.'
  )
  sagemaker_port: int | None = pydantic.Field(
    default=None, description='The port for the server to listen on for Sagemaker requests. Default is 8080.'
  )
  sagemaker_safe_port_range: str | None = pydantic.Field(
    default=None, description='Set the allowed port range for endpoints other than the SageMaker endpoints.'
  )
  sagemaker_thread_count: int | None = pydantic.Field(
    default=None, description='Number of threads handling Sagemaker requests. Default is 8.'
  )
  allow_vertex_ai: bool | None = pydantic.Field(
    default=None,
    description='Allow the server to listen for Vertex AI requests. Default is true if AIP_MODE=PREDICTION, false otherwise.',
  )
  vertex_ai_port: int | None = pydantic.Field(
    default=None,
    description='The port for the server to listen on for Vertex AI requests. Default is AIP_HTTP_PORT if set, 8080 otherwise.',
  )
  vertex_ai_thread_count: int | None = pydantic.Field(
    default=None, description='Number of threads handling Vertex AI requests. Default is 8.'
  )
  vertex_ai_default_model: str | None = pydantic.Field(
    default=None, description='The name of the model to use for single-model inference requests.'
  )
  metrics_address: str | None = pydantic.Field(
    default=None,
    description='The address for the metrics server to bind to. Default is the same as --http-address if built with HTTP support. Otherwise, default is 0.0.0.0',
  )
  metrics_interval_ms: float | None = pydantic.Field(
    default=None,
    description='Metrics will be collected once every <metrics-interval-ms> milliseconds. Default is 2000 milliseconds.',
  )
  metrics_config: list[str] | None = pydantic.Field(
    default=None,
    description='Specify a metrics-specific configuration setting. The format of this flag is --metrics-config=<setting>=<value>. It can be specified multiple times.',
  )
  trace_config: list[str] | None = pydantic.Field(
    default=None,
    description='Specify global or trace mode specific configuration setting. The format of this flag is --trace-config <mode>,<setting>=<value>. Where <mode> is either "triton" or "opentelemetry". The default is "triton". To specify global trace settings (level, rate, count, or mode), the format would be --trace-config <setting>=<value>. For "triton" mode, the server will use Triton\'s Trace APIs. For "opentelemetry" mode, the server will use OpenTelemetry\'s APIs to generate, collect and export traces for individual inference requests.',
  )
  backend_directory: str | None = pydantic.Field(
    default=None,
    description="The global directory searched for backend shared libraries. Default is '/opt/tritonserver/backends'.",
  )
  backend_config: list[str] | None = pydantic.Field(
    default=None,
    description="Specify a backend-specific configuration setting. The format of this flag is --backend-config=<backend_name>,<setting>=<value>. Where <backend_name> is the name of the backend, such as 'tensorrt'.",
  )
  repoagent_directory: str | None = pydantic.Field(
    default=None,
    description="The global directory searched for repository agent shared libraries. Default is '/opt/tritonserver/repoagents'.",
  )
  cache_config: list[str] | None = pydantic.Field(
    default=None,
    description="Specify a cache-specific configuration setting. The format of this flag is --cache-config=<cache_name>,<setting>=<value>. Where <cache_name> is the name of the cache, such as 'local' or 'redis'. Example: --cache-config=local,size=1048576 will configure a 'local' cache implementation with a fixed buffer pool of size 1048576 bytes.",
  )
  cache_directory: str | None = pydantic.Field(
    default=None,
    description="The global directory searched for cache shared libraries. Default is '/opt/tritonserver/caches'. This directory is expected to contain a cache implementation as a shared library with the name 'libtritoncache.so'.",
  )
  rate_limit: str | None = pydantic.Field(
    default=None,
    description='Specify the mode for rate limiting. Options are "execution_count" and "off". The default is "off". For "execution_count", the server will determine the instance using configured priority and the number of time the instance has been used to run inference. The inference will finally be executed once the required resources are available. For "off", the server will ignore any rate limiter config and run inference as soon as an instance is ready.',
  )
  rate_limit_resource: list[str] | None = pydantic.Field(
    default=None,
    description='The number of resources available to the server. The format of this flag is --rate-limit-resource=<resource_name>:<count>:<device>. The <device> is optional and if not listed will be applied to every device. If the resource is specified as "GLOBAL" in the model configuration the resource is considered shared among all the devices in the system. The <device> property is ignored for such resources. This flag can be specified multiple times to specify each resources and their availability. By default, the max across all instances that list the resource is selected as its availability. The values for this flag is case-insensitive.',
  )
  pinned_memory_pool_byte_size: int | None = pydantic.Field(
    default=None,
    description="The total byte size that can be allocated as pinned system memory. If GPU support is enabled, the server will allocate pinned system memory to accelerate data transfer between host and devices until it exceeds the specified byte size. If 'numa-node' is configured via --host-policy, the pinned system memory of the pool size will be allocated on each numa node. This option will not affect the allocation conducted by the backend frameworks. Default is 256 MB.",
  )
  cuda_memory_pool_byte_size: list[str] | None = pydantic.Field(
    default=None,
    description='The total byte size that can be allocated as CUDA memory for the GPU device. If GPU support is enabled, the server will allocate CUDA memory to minimize data transfer between host and devices until it exceeds the specified byte size. This option will not affect the allocation conducted by the backend frameworks. The argument should be 2 integers separated by colons in the format <GPU device ID>:<pool byte size>. This option can be used multiple times, but only once per GPU device. Subsequent uses will overwrite previous uses for the same GPU device. Default is 64 MB.',
  )
  cuda_virtual_address_size: list[str] | None = pydantic.Field(
    default=None,
    description='The total CUDA virtual address size that will be used for each implicit state when growable memory is used. This value determines the maximum size of each implicit state. The state size cannot go beyond this value. The argument should be 2 integers separated by colons in the format <GPU device ID>:<CUDA virtual address size>. This option can be used multiple times, but only once per GPU device. Subsequent uses will overwrite previous uses for the same GPU device. Default is 1 GB.',
  )
  min_supported_compute_capability: float | None = pydantic.Field(
    default=None,
    description="The minimum supported CUDA compute capability. GPUs that don't support this compute capability will not be used by the server.",
  )
  buffer_manager_thread_count: int | None = pydantic.Field(
    default=None,
    description='The number of threads used to accelerate copies and other operations required to manage input and output tensor contents. Default is 0.',
  )
  host_policy: list[str] | None = pydantic.Field(
    default=None,
    description="Specify a host policy setting associated with a policy name. The format of this flag is --host-policy=<policy_name>,<setting>=<value>. Currently supported settings are 'numa-node', 'cpu-cores'. Note that 'numa-node' setting will affect pinned memory pool behavior, see --pinned-memory-pool for more detail.",
  )
  model_load_gpu_limit: str | None = pydantic.Field(
    default=None,
    description='Specify the limit on GPU memory usage as a fraction. If model loading on the device is requested and the current memory usage exceeds the limit, the load will be rejected. If not specified, the limit will not be set.',
  )

  @pydantic.model_serializer(when_used='always', mode='wrap')
  def serialize_to_cli_args(self, nxt: pydantic.SerializerFunctionWrapHandler) -> list[str]:
    args = []
    dumped_data = {k: v for k, v in nxt(self).items() if v is not None}

    for field_name, value in dumped_data.items():
      field_info = self.__class__.model_fields[field_name]
      cli_arg = f'--{field_name.replace("_", "-")}'

      annotation = field_info.annotation
      is_list = False
      core_type = annotation

      # Handle Optional[T] and Optional[List[T]]
      if annotation and hasattr(annotation, '__origin__') and annotation.__origin__ is typing.Union:
        non_none_args = [arg for arg in getattr(annotation, '__args__', []) if arg is not type(None)]
        if len(non_none_args) == 1:
          core_type = non_none_args[0]

      # Check if the core type is List[T]
      if core_type and hasattr(core_type, '__origin__') and core_type.__origin__ is list:
        is_list = True
        list_args = getattr(core_type, '__args__', [])
        core_type = list_args[0] if list_args else typing.Any

      if core_type is bool:
        args.append(f'{cli_arg}={str(value).lower()}')
      elif is_list and isinstance(value, list):
        args.extend([f'{cli_arg}={item}' for item in value])
      else:
        args.append(f'{cli_arg}={value}')

    return args

  if typing.TYPE_CHECKING:

    def model_dump(
      self,
      *,
      mode: typing.Literal['json', 'python'] | str = ...,
      context: typing.Any = ...,
      serialize_as_any: bool = ...,
      fallback: typing.Any = ...,
      include: typing.Any = ...,
      exclude: typing.Any = ...,
      by_alias: bool | None = False,
      exclude_unset: bool = False,
      exclude_defaults: bool = False,
      exclude_none: bool = True,
      round_trip: bool = False,
      warnings: bool | typing.Literal['none', 'warn', 'error'] = True,
    ) -> list[str]: ...


InferParameter = typing.Union[pydantic.StrictFloat, pydantic.StrictInt, pydantic.StrictBool, str]
Parameters = dict[str, InferParameter]


class RequestInput(pydantic.BaseModel):
  """RequestInput Model

  $request_input =
  {
    "name" : $string,
    "shape" : [ $number, ... ],
    "datatype"  : $string,
    "parameters" : $parameters #optional,
    "data" : $tensor_data
  }
  """

  name: str
  shape: list[int]
  datatype: str
  parameters: Parameters | None = None
  data: list[typing.Any]


class RequestOutput(pydantic.BaseModel):
  """RequestOutput Model

  $request_output =
  {
    "name" : $string,
    "parameters" : $parameters #optional,
  }
  """

  name: str
  parameters: Parameters | None = None


class InferenceRequest(pydantic.BaseModel):
  inputs: list[RequestInput]
  id: str | None = pydantic.Field(default=None)
  parameters: dict[str, typing.Any] | None = pydantic.Field(default=None)
  outputs: list[dict[str, typing.Any]] | None = pydantic.Field(default=None)


bento_args = bentoml.use_arguments(BentoArgs)


kserve_app = fastapi.FastAPI()


@bentoml.asgi_app(kserve_app, path='/v2')
@bentoml.service(
  name='bentotriton-service',
  resources={'gpu': 1, 'gpu_type': 'nvidia-a100-80gb'},
  image=bentoml.images.Image(base_image='nvcr.io/nvidia/tritonserver:25.03-py3').pyproject_toml('pyproject.toml'),
)
class Triton:
  @bentoml.on_startup
  async def run_server(self):
    command = ['tritonserver', *bento_args.model_dump()]
    logger.debug('tritonserver commands: %s', command)
    self.process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)

    await asyncio.sleep(1)  # Give the process a brief moment to potentially fail

    return_code = self.process.poll()
    if return_code is not None:
      raise RuntimeError(
        f'Triton server process terminated unexpectedly shortly after starting with return code {return_code}'
      )

    logger.debug('Triton server process started.')
    self.client = httpx.AsyncClient(base_url=f'http://{bento_args.http_address}:{bento_args.http_port}/v2')

  @bentoml.on_shutdown
  async def shutdown_server(self):
    self.process.terminate()
    await self.client.aclose()

  async def __is_ready__(self) -> bool:
    resp = await self.client.get('/health/ready')
    return resp.status_code == 200

  async def __is_alive__(self) -> bool:
    resp = await self.client.get('/health/live')
    return resp.status_code == 200

  @bentoml.on_startup
  def mount_metadata_routers(self):
    router = fastapi.APIRouter(prefix='/health')
    ENDPOINTS = [('/live', self.triton_health_live, ['GET']), ('/ready', self.triton_health_ready, ['GET'])]
    for route, endpoint, methods in ENDPOINTS:
      router.add_api_route(route, endpoint=endpoint, methods=methods, include_in_schema=True)
    kserve_app.include_router(router)

  async def triton_health_live(self):
    return await self.client.get('/health/live')

  async def triton_health_ready(self):
    return await self.client.get('/health/ready')

  @bentoml.on_startup
  def mount_models_routers(self):
    router = fastapi.APIRouter(prefix='/models')
    ENDPOINTS: list[tuple[str, typing.Callable[..., typing.Any], list[str], type[pydantic.BaseModel] | None]] = [
      ('/{model_name}/ready', self.triton_model_ready, ['GET'], None),
      ('/{model_name}/versions/{model_version}/ready', self.triton_model_with_version_ready, ['GET'], None),
      ('/{model_name}', self.triton_model_metadata, ['GET'], None),
      ('/{model_name}/versions/{model_version}', self.triton_model_with_version_metadata, ['GET'], None),
      ('/{model_name}/infer', self.triton_model_infer, ['PUT'], None),
      ('/{model_name}/versions/{model_version}/infer', self.triton_model_with_version_infer, ['PUT'], None),
    ]
    for route, endpoint, methods, response_model in ENDPOINTS:
      router.add_api_route(
        route, endpoint=endpoint, methods=methods, include_in_schema=True, response_model=response_model
      )
    kserve_app.include_router(router)

  async def triton_model_ready(self, model_name: str):
    resp = await self.client.get(f'/models/{model_name}/ready')
    if resp.status_code != 200:
      raise HTTPException(status_code=resp.status_code, detail=f"Model '{model_name}' is not ready")
    return PlainTextResponse('\n')

  async def triton_model_with_version_ready(self, model_name: str, model_version: str):
    resp = await self.client.get(f'/models/{model_name}/versions/{model_version}/ready')
    if resp.status_code != 200:
      raise HTTPException(
        status_code=resp.status_code, detail=f"Model '{model_name}' version '{model_version}' is not ready"
      )
    return PlainTextResponse('\n')

  async def triton_model_metadata(self, model_name: str):
    resp = await self.client.get(f'/models/{model_name}')
    if resp.status_code != 200:
      raise HTTPException(status_code=resp.status_code, detail=f"Failed to retrieve '{model_name}' metadata")
    return JSONResponse(resp.json())

  async def triton_model_with_version_metadata(self, model_name: str, model_version: str):
    resp = await self.client.get(f'/models/{model_name}/versions/{model_version}')
    if resp.status_code != 200:
      raise HTTPException(status_code=resp.status_code, detail=f"Failed to retrieve '{model_name}' metadata")
    return JSONResponse(resp.json())

  async def triton_model_infer(self, model_name: str, request: InferenceRequest):
    resp = await self.client.post(
      f'/models/{model_name}/infer', json=request.model_dump(exclude_none=True, mode='json')
    )
    if resp.status_code != 200:
      raise HTTPException(
        status_code=resp.status_code, detail=f"Failed to infer '{model_name}': {resp.json()['error']}"
      )
    return JSONResponse(resp.json())

  async def triton_model_with_version_infer(self, model_name: str, model_version: str, request: InferenceRequest):
    resp = await self.client.post(
      f'/models/{model_name}/versions/{model_version}/infer', json=request.model_dump(exclude_none=True, mode='json')
    )
    if resp.status_code != 200:
      raise HTTPException(
        status_code=resp.status_code,
        detail=f"Failed to infer '{model_name}' version '{model_version}': {resp.json()['error']}",
      )
    return JSONResponse(resp.json())
