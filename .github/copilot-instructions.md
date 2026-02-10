## Quick task receipt

This file helps a code-writing agent onboard the pyFade repository quickly. It contains a concise summary of the project, reproducible environment and build/run/validation steps tested for this repository, a short layout map pointing to the important files, and common pitfalls to avoid. Trust these instructions; only search the repository if something here is missing or clearly out of date.

---

## 1) High-level summary

`InferenceGate` is a Python library designed to facilitate efficient and convenient AI inference replay for testing, debugging, and development purposes. It provides tools to capture, store, and replay AI model inferences, enabling developers to simulate various scenarios and validate model behavior without the need for repeated live inferences. This is particularly useful for debugging complex AI systems, conducting regression tests, and optimizing model performance in a controlled environment.

At it's core it is a HTTP proxy server that accepts OpenAI-compatible Responses API and Chat Completions API requests, then routes them to either a real AI model endpoint or to a local replay database, based on user-defined rules and configurations.

Additionally, there's an option to filter and block requests based on keywords or patterns, enhancing security and compliance during inference operations.

## 2) Key Principles
- **Async:** The codebase is designed to be fully asynchronous, leveraging `asyncio` for concurrency. `aiohttp` is used for HTTP server and client functionality.

## 3) Code style
- Follows PEP 8 style guidelines.
- **ALWAYS** use 4 spaces for indentation. **NEVER** use tabs.
- Type hints are used extensively.
- Keep code documented with docstrings and comments. **NEVER** remove comments, instead update them if they are out of date.
- Line length limit is 140 characters. Do not split function signatures or calls across multiple lines unless absolutely necessary.
- Use `yapf` for code formatting, configured in `pyproject.toml`. Run `yapf -i <file>` to format a file in place. Consider `yapf` to be the source of truth for formatting. Run it on changed files before committing.
- All modules, classes, and functions should have docstrings. Docstrings should start on new line after triple double quotes. Docstrings for class should describe the purpose of the class and any important details. Docstrings for methods should describe the purpose of the method, its parameters, return values, and any important details. Test docstrings should describe what is being tested and the expected outcome.
- Use `logging` module for logging, do not use print statements. Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Use per-class loggers via `self.log = logging.getLogger(CLASS_NAME)`. Use lazy evaluation of log messages via `%` of the logger methods, e.g. `logger.debug("Message: %s", variable)` whenever possible.
- Do not use local imports unless absolutely necessary to avoid circular dependencies, prefer module-level imports.
- Use f-strings for string formatting, except in logging calls where lazy evaluation via `%` is preferred.
- UI classes should have separate setup_ui() method that builds the UI components and set_XXX() methods for setting data and populating the UI with values.
- UI set_XXX() methods should branch on single if statement to handle None/new object vs existing object, one branch for new object with empty/default values, another branch for existing object with real values.
- SQLAlchemy ORM is used for database models and queries, orchestrated via DatasetDatabase class.
- Following Google Material Design principles for UI layout and behavior, using qt_material for theming.

Example of correct code style:
```python
"""
`ModuleName` is providing implementation for XYZ.

Key classes: `ClassName1`, `ClassName2`
"""
class ClassName1:
    """
    ClassName1 does XYZ.
    """
    attribute1: int
    attribute2: str

    def method_with_lots_of_kwargs(self, param1: int, param2: str, kwarg_param3: str = "default", kwarg_param4: int = 42,
                                   kwarg_param5: float = 3.14, kwarg_param6: bool = True) -> bool:
        """
        Do something important.

        Uses `param1` to do X and `param2` to do Y.
        Optional `kwarg_param3` controls Z behavior.

        Returns the return value.
        """
        # First, log the method call with parameters
        self.log.debug("Executing method_with_lots_of_kwargs with param1=%d, param2=%s", param1, param2)
        # Method implementation here
        return True
```

## 4) Documentation

- All public classes and methods should have docstrings. Docstrings should start on new line after triple double quotes. Docstrings for class should describe the purpose of the class and any important details. Docstrings for methods should describe the purpose of the method, its parameters, return values, and any important details. Test docstrings should describe what is being tested and the expected outcome.
- The `docs/` directory contains markdown files with documentation for the project. The `docs/modules/` directory contains module-specific documentation. The `docs/cli.md` file contains documentation for the command-line interface (CLI). The `docs/index.md` file contains the main documentation index.
- `docs/architecture.md` contains an overview of the system architecture and design principles. It describes the main components of the system and how they interact with each other.
- The documentation should be kept up to date with the code. When making changes to the code, check if the documentation needs to be updated as well. If you see any outdated documentation, please update it.
- Keep implementation specifics in the `docs/modules/` directory, and keep high-level documentation in the `docs/` directory. The `docs/index.md` file should contain an index of all documentation files.

## 5) Building and running

### Backend
- Install the package: `pip install -e .`
- Run the CLI: `python -m inference_gate.cli --help`
- Start the proxy server with WebUI: `python -m inference_gate.cli --webui-port 8081`

### WebUI (Frontend)
The WebUI is a React-based single-page application (SPA) built with modern web technologies:

**Tech Stack:**
- React 19 with TypeScript
- Vite for build tooling
- Tailwind CSS v4 for styling
- shadcn/ui component library (Radix UI primitives)
- React Router for client-side routing

**Development:**
```bash
cd webui-frontend
npm install
npm run dev    # Start dev server on http://localhost:5173
```

**Building:**
```bash
cd webui-frontend
npm run build  # Outputs to ../src/inference_gate/webui/static/
```

**Structure:**
- `webui-frontend/src/components/` - Reusable React components
- `webui-frontend/src/components/ui/` - shadcn/ui component library
- `webui-frontend/src/pages/` - Page-level components (Dashboard, CacheList, EntryDetail)
- `webui-frontend/src/api/` - API client for backend communication
- Built assets are served by the Python backend's aiohttp server

**Code Style:**
- Use TypeScript with strict type checking
- Functional components with hooks (no class components)
- Use default exports for pages/components (e.g., Layout, CacheTable, Dashboard) to match existing codebase patterns
- Path alias `@/` points to `src/` directory
- Use shadcn/ui components instead of raw Tailwind classes
- Use Lucide React icons for visual elements
- Follow existing component patterns (Card, Badge, Button, etc.)

**Design System:**
- shadcn/ui provides consistent design tokens via CSS variables
- Color palette defined in `src/index.css` using HSL values
- Use semantic color names: `background`, `foreground`, `primary`, `secondary`, `muted`, `destructive`, etc.
- Responsive design with mobile-first approach
- Hover effects and transitions for interactive elements

## 6) Tests and validation

- Unit tests are located in the `tests/` directory. Each module should have a corresponding test file, e.g. `module_name.py` should have `test_module_name.py`.
- Tests should be run using `pytest`. Run `pytest tests/` to run all tests. Run `pytest tests/test_module_name.py` to run tests for a specific module.
- Tests should be kept up to date with the code. When making changes to the code, check if the tests need to be updated as well. If you see any outdated tests, please update them.
- Test docstrings should describe what is being tested and the expected outcome. Test names should be descriptive of what is being tested, e.g. `test_method_name_behavior_under_condition`.
- Use fixtures for setting up test data and state. Use `pytest` fixtures to create reusable test setup code. Fixtures should be defined in `tests/conftest.py` and imported into test files as needed.
- Aim for high test coverage, but prioritize meaningful tests that cover important functionality and edge cases over achieving 100% coverage. Use code coverage tools to identify untested code paths and add tests as needed.
