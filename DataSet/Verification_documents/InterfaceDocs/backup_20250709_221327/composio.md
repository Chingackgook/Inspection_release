# ComposioToolSet API Documentation

## Class: ComposioToolSet

### Initialization

#### __init__

```python
def __init__(
    api_key: t.Optional[str] = None,
    base_url: t.Optional[str] = None,
    entity_id: str = DEFAULT_ENTITY_ID,
    workspace_id: t.Optional[str] = None,
    workspace_config: t.Optional[WorkspaceConfigType] = None,
    metadata: t.Optional[MetadataType] = None,
    processors: t.Optional[ProcessorsType] = None,
    logging_level: LogLevel = LogLevel.INFO,
    output_dir: t.Optional[Path] = None,
    output_in_file: bool = False,
    verbosity_level: t.Optional[int] = None,
    allow_tracing: bool = False,
    connected_account_ids: t.Optional[t.Dict[AppType, str]] = None,
    *,
    max_retries: int = 3,
    lockfile: t.Optional[Path] = None,
    lock: bool = True,
    **kwargs: t.Any,
) -> None:
```

- **Parameters:**
  - `api_key`: Optional API key for Composio.
  - `base_url`: Optional base URL for the Composio API server.
  - `entity_id`: ID of the entity to execute the action on. Defaults to "default".
  - `workspace_id`: Optional workspace ID for loading an existing workspace.
  - `workspace_config`: Optional configuration for the workspace.
  - `metadata`: Optional additional metadata for executing actions (JSON serializable).
  - `processors`: Optional request and response processors.
  - `logging_level`: Logging level for the toolset.
  - `output_dir`: Optional directory for output files.
  - `output_in_file`: Boolean indicating if output should be written to a file.
  - `verbosity_level`: Optional verbosity level for logs.
  - `allow_tracing`: Boolean indicating if tracing is allowed.
  - `connected_account_ids`: Optional connected account IDs.
  - `max_retries`: Maximum number of retries for action execution.
  - `lockfile`: Optional path for version lock file.
  - `lock`: Boolean indicating if version locking is enabled.
  - `**kwargs`: Additional optional parameters.

- **Returns:** None

- **Example:**
```python
toolset = ComposioToolSet(api_key='your_api_key', base_url='https://api.composio.com')
```

### Properties

#### api_key

- **Type:** str
- **Returns:** API key for the Composio toolset.
- **Raises:** `ApiKeyNotProvidedError` if API key is not set.

#### client

- **Type:** Composio
- **Returns:** Initializes and returns the Composio client.

#### workspace

- **Type:** Workspace
- **Returns:** Workspace for this toolset instance.

### Methods

#### check_connected_account

```python
def check_connected_account(self, action: ActionType, entity_id: t.Optional[str] = None) -> None:
```

- **Parameters:**
  - `action`: Action to execute for checking connected account.
  - `entity_id`: Optional ID of the entity to check against.

- **Returns:** None
- **Raises:** `ConnectedAccountNotFoundError` if no connected account found.

- **Example:**
```python
toolset.check_connected_account(action=Action.SOME_ACTION)
```

#### execute_action

```python
def execute_action(
    self,
    action: ActionType,
    params: dict,
    metadata: t.Optional[t.Dict] = None,
    entity_id: t.Optional[str] = None,
    connected_account_id: t.Optional[str] = None,
    text: t.Optional[str] = None,
    *,
    processors: t.Optional[ProcessorsType] = None,
    _check_requested_actions: bool = False,
) -> t.Dict:
```

- **Parameters:**
  - `action`: Action to execute.
  - `params`: Parameters for the action.
  - `metadata`: Optional metadata for local action.
  - `entity_id`: Optional entity ID for executing the action.
  - `connected_account_id`: Optional connection ID for executing the remote action.
  - `text`: Optional text for generating function calling metadata.
  - `processors`: Optional processors for request and response.
  - `_check_requested_actions`: Boolean to check if requested actions are valid.

- **Returns:** Output object from the function call (dict).

- **Example:**
```python
response = toolset.execute_action(action=Action.SOME_ACTION, params={"param1": "value"})
```

#### execute_request

```python
def execute_request(
    self,
    endpoint: str,
    method: str,
    *,
    body: t.Optional[t.Dict] = None,
    parameters: t.Optional[t.List[CustomAuthParameter]] = None,
    connection_id: t.Optional[str] = None,
    app: t.Optional[AppType] = None,
) -> t.Dict:
```

- **Parameters:**
  - `endpoint`: API endpoint to call.
  - `method`: HTTP method (GET, POST, etc.).
  - `body`: Optional request body data.
  - `parameters`: Optional additional auth parameters.
  - `connection_id`: Optional ID of the connected account.
  - `app`: Optional App type for connection lookup.

- **Returns:** Response from the proxy request (dict).
- **Raises:** `InvalidParams` if both `connection_id` and `app` are missing.

- **Example:**
```python
response = toolset.execute_request(endpoint='/some/endpoint', method='POST', body={"key": "value"})
```

#### validate_tools

```python
def validate_tools(
    self,
    apps: t.Optional[t.Sequence[AppType]] = None,
    actions: t.Optional[t.Sequence[ActionType]] = None,
    tags: t.Optional[t.Sequence[TagType]] = None,
) -> None:
```

- **Parameters:**
  - `apps`: Optional sequence of app types.
  - `actions`: Optional sequence of action types.
  - `tags`: Optional sequence of tags.

- **Returns:** None

- **Example:**
```python
toolset.validate_tools(apps=[App.APP1], actions=[Action.ACTION1])
```

#### get_action_schemas

```python
def get_action_schemas(
    self,
    apps: t.Optional[t.Sequence[AppType]] = None,
    actions: t.Optional[t.Sequence[ActionType]] = None,
    tags: t.Optional[t.Sequence[TagType]] = None,
    *,
    check_connected_accounts: bool = True,
    _populate_requested: bool = False,
) -> t.List[ActionModel]:
```

- **Parameters:**
  - `apps`: Optional sequence of app types.
  - `actions`: Optional sequence of action types.
  - `tags`: Optional sequence of tags.
  - `check_connected_accounts`: Boolean to check connected accounts.
  - `_populate_requested`: Boolean to populate requested actions.

- **Returns:** List of action schemas (list of `ActionModel`).

- **Example:**
```python
schemas = toolset.get_action_schemas(apps=[App.APP1], actions=[Action.ACTION1])
```

#### create_trigger_listener

```python
def create_trigger_listener(self, timeout: float = 15.0) -> TriggerSubscription:
```

- **Parameters:**
  - `timeout`: Optional timeout for trigger subscription.

- **Returns:** TriggerSubscription object.

- **Example:**
```python
listener = toolset.create_trigger_listener(timeout=10)
```

This documentation provides an overview of the `ComposioToolSet` class, including its initialization parameters, properties, and methods, along with detailed usage examples for each.