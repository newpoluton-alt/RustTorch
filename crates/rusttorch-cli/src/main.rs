use std::{
    env,
    ffi::OsString,
    fmt, fs,
    io::Write,
    path::{Path, PathBuf},
    process::{self, Command},
};

use tempfile::NamedTempFile;
use toml_edit::{DocumentMut, Item, value};

const HELP: &str = "RustTorch managed LibTorch setup

Usage:
  rusttorch setup --backend auto
  rusttorch setup --backend cpu
  rusttorch setup --backend cuda-12.6
  rusttorch --help
  rusttorch --version";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendRequest {
    Auto,
    Cpu,
    Cuda126,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedBackend {
    Preconfigured,
    Cpu,
    Cuda126,
}

impl fmt::Display for ResolvedBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Preconfigured => "preconfigured",
            Self::Cpu => "cpu",
            Self::Cuda126 => "cuda-12.6",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CliAction {
    Help,
    Version,
    Setup(BackendRequest),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Platform {
    Linux,
    Windows,
    Macos,
    Other,
}

impl Platform {
    fn from_os(os: &str) -> Self {
        match os {
            "linux" => Self::Linux,
            "windows" => Self::Windows,
            "macos" => Self::Macos,
            _ => Self::Other,
        }
    }

    fn current() -> Self {
        Self::from_os(env::consts::OS)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DriverVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

impl DriverVersion {
    const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct CliError(String);

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

fn parse_backend(value: &str) -> Result<BackendRequest, CliError> {
    match value {
        "auto" => Ok(BackendRequest::Auto),
        "cpu" => Ok(BackendRequest::Cpu),
        "cuda-12.6" => Ok(BackendRequest::Cuda126),
        _ => Err(CliError::new(
            "backend must be one of: auto, cpu, cuda-12.6",
        )),
    }
}

fn parse_args<I, S>(args: I) -> Result<CliAction, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    match args.as_slice() {
        [] => Ok(CliAction::Help),
        [arg] if arg == "--help" || arg == "-h" => Ok(CliAction::Help),
        [arg] if arg == "--version" || arg == "-V" => Ok(CliAction::Version),
        [setup, backend, value] if setup == "setup" && backend == "--backend" => parse_backend(
            value
                .to_str()
                .ok_or_else(|| CliError::new("backend must be valid UTF-8"))?,
        )
        .map(CliAction::Setup),
        _ => Err(CliError::new(format!("invalid arguments\n\n{HELP}"))),
    }
}

fn is_libtorch_preconfigured(mut is_set: impl FnMut(&str) -> bool) -> bool {
    [
        "LIBTORCH_USE_PYTORCH",
        "LIBTORCH",
        "LIBTORCH_INCLUDE",
        "LIBTORCH_LIB",
    ]
    .into_iter()
    .any(&mut is_set)
}

fn is_setup_preconfigured(
    platform: Platform,
    is_set: impl FnMut(&str) -> bool,
    mut path_exists: impl FnMut(&Path) -> bool,
) -> bool {
    is_libtorch_preconfigured(is_set)
        || (platform == Platform::Linux && path_exists(Path::new("/usr/lib/libtorch.so")))
}

fn parse_driver_output(output: &str) -> Result<DriverVersion, CliError> {
    let version = output
        .lines()
        .next()
        .ok_or_else(|| CliError::new("nvidia-smi returned no driver version"))?;
    let mut components = version.trim().split('.');
    let major = parse_driver_component(components.next())?;
    let minor = parse_driver_component(components.next())?;
    let patch = components
        .next()
        .map(|value| value.parse())
        .transpose()
        .map_err(|_| CliError::new("nvidia-smi returned an invalid driver version"))?
        .unwrap_or(0);
    if components.next().is_some() {
        return Err(CliError::new(
            "nvidia-smi returned an invalid driver version",
        ));
    }
    Ok(DriverVersion::new(major, minor, patch))
}

fn parse_driver_component(component: Option<&str>) -> Result<u32, CliError> {
    component
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| CliError::new("nvidia-smi returned an invalid driver version"))
}

fn minimum_cuda_driver(platform: Platform) -> Option<DriverVersion> {
    match platform {
        Platform::Linux => Some(DriverVersion::new(525, 60, 13)),
        Platform::Windows => Some(DriverVersion::new(528, 33, 0)),
        Platform::Macos | Platform::Other => None,
    }
}

fn resolve_backend(
    request: BackendRequest,
    platform: Platform,
    configured: bool,
    driver: Option<DriverVersion>,
) -> Result<ResolvedBackend, CliError> {
    if configured {
        return match request {
            BackendRequest::Auto => Ok(ResolvedBackend::Preconfigured),
            BackendRequest::Cpu | BackendRequest::Cuda126 => Err(CliError::new(
                "explicit backends conflict with an active LibTorch installation or environment",
            )),
        };
    }

    match request {
        BackendRequest::Auto => Ok(match minimum_cuda_driver(platform) {
            Some(minimum) if driver.is_some_and(|version| version >= minimum) => {
                ResolvedBackend::Cuda126
            }
            _ => ResolvedBackend::Cpu,
        }),
        BackendRequest::Cpu => Ok(ResolvedBackend::Cpu),
        BackendRequest::Cuda126 => {
            let minimum = minimum_cuda_driver(platform)
                .ok_or_else(|| CliError::new("cuda-12.6 is supported only on Linux and Windows"))?;
            let driver = driver.ok_or_else(|| CliError::new("no NVIDIA driver was detected"))?;
            if driver < minimum {
                return Err(CliError::new("the NVIDIA driver is too old for CUDA 12.x"));
            }
            Ok(ResolvedBackend::Cuda126)
        }
    }
}

fn resolve_setup_backend(
    request: BackendRequest,
    platform: Platform,
    configured: bool,
    detect: impl FnOnce() -> Option<DriverVersion>,
) -> Result<ResolvedBackend, CliError> {
    let needs_driver = !configured
        && matches!(platform, Platform::Linux | Platform::Windows)
        && matches!(request, BackendRequest::Auto | BackendRequest::Cuda126);
    resolve_backend(
        request,
        platform,
        configured,
        needs_driver.then(detect).flatten(),
    )
}

fn detect_driver() -> Option<DriverVersion> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_driver_output(std::str::from_utf8(&output.stdout).ok()?).ok()
}

#[derive(Debug, PartialEq, Eq)]
enum TorchCudaEnvironment {
    Inherit,
    Remove,
    Set(OsString),
}

#[derive(Debug, PartialEq, Eq)]
struct CargoCheckSpec {
    program: OsString,
    args: Vec<OsString>,
    current_dir: PathBuf,
    torch_cuda_environment: TorchCudaEnvironment,
}

fn backend_name(backend: ResolvedBackend) -> Option<&'static str> {
    match backend {
        ResolvedBackend::Preconfigured => None,
        ResolvedBackend::Cpu => Some("cpu"),
        ResolvedBackend::Cuda126 => Some("cuda-12.6"),
    }
}

fn target_directory(backend: ResolvedBackend) -> Option<&'static str> {
    match backend {
        ResolvedBackend::Preconfigured => None,
        ResolvedBackend::Cpu => Some("target/rusttorch/cpu"),
        ResolvedBackend::Cuda126 => Some("target/rusttorch/cuda-12.6"),
    }
}

fn nested_item<'a>(document: &'a DocumentMut, table: &str, key: &str) -> Option<&'a Item> {
    document.get(table)?.get(key)
}

fn ensure_project_root(root: &Path) -> Result<(), CliError> {
    if !root.join("Cargo.toml").is_file() {
        return Err(CliError::new(format!(
            "no Cargo.toml found in {}",
            root.display()
        )));
    }
    Ok(())
}

fn configure_project(root: &Path, backend: ResolvedBackend) -> Result<PathBuf, CliError> {
    ensure_project_root(root)?;
    let config_directory = root.join(".cargo");
    let config_path = config_directory.join("config.toml");
    if backend == ResolvedBackend::Preconfigured {
        return Ok(config_path);
    }

    let original = match fs::read_to_string(&config_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(CliError::new(format!(
                "could not read {}: {error}",
                config_path.display()
            )));
        }
    };
    let mut document = if original.is_empty() {
        DocumentMut::new()
    } else {
        original.parse::<DocumentMut>().map_err(|error| {
            CliError::new(format!(
                "could not parse {}: {error}",
                config_path.display()
            ))
        })?
    };

    for table in ["build", "env"] {
        if document
            .get(table)
            .is_some_and(|item| item.as_table_like().is_none())
        {
            return Err(CliError::new(format!(
                "cannot update {table}: it is not a TOML table"
            )));
        }
    }

    let marker = nested_item(&document, "env", "RUSTTORCH_BACKEND")
        .and_then(Item::as_str)
        .and_then(|value| match value {
            "cpu" => Some(ResolvedBackend::Cpu),
            "cuda-12.6" => Some(ResolvedBackend::Cuda126),
            _ => None,
        });
    let configured_target = nested_item(&document, "build", "target-dir");
    let owns_configuration = marker.is_some_and(|old_backend| {
        configured_target.and_then(Item::as_str) == target_directory(old_backend)
    });

    if configured_target.is_some() && !owns_configuration {
        return Err(CliError::new(
            "build.target-dir is user-owned; remove it or choose the RustTorch-managed value manually",
        ));
    }
    if nested_item(&document, "env", "RUSTTORCH_BACKEND").is_some() && !owns_configuration {
        return Err(CliError::new(
            "RUSTTORCH_BACKEND is not paired with a RustTorch-managed target directory",
        ));
    }
    if nested_item(&document, "env", "TORCH_CUDA_VERSION").is_some() && !owns_configuration {
        return Err(CliError::new(
            "TORCH_CUDA_VERSION is user-owned; remove it before RustTorch setup changes backends",
        ));
    }

    document["build"]["target-dir"] = value(target_directory(backend).unwrap());
    document["env"]["RUSTTORCH_BACKEND"] = value(backend_name(backend).unwrap());
    match backend {
        ResolvedBackend::Cuda126 => {
            document["env"]["TORCH_CUDA_VERSION"] = value("cu126");
        }
        ResolvedBackend::Cpu => {
            document["env"]
                .as_table_like_mut()
                .unwrap()
                .remove("TORCH_CUDA_VERSION");
        }
        ResolvedBackend::Preconfigured => unreachable!(),
    }

    fs::create_dir_all(&config_directory).map_err(|error| {
        CliError::new(format!(
            "could not create {}: {error}",
            config_directory.display()
        ))
    })?;
    let mut temporary = NamedTempFile::new_in(&config_directory).map_err(|error| {
        CliError::new(format!(
            "could not create a temporary Cargo configuration: {error}"
        ))
    })?;
    temporary
        .write_all(document.to_string().as_bytes())
        .and_then(|()| temporary.flush())
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| CliError::new(format!("could not write Cargo configuration: {error}")))?;
    temporary.persist(&config_path).map_err(|error| {
        CliError::new(format!(
            "could not replace {} atomically: {}",
            config_path.display(),
            error.error
        ))
    })?;
    Ok(config_path)
}

fn cargo_check_spec(root: &Path, backend: ResolvedBackend) -> CargoCheckSpec {
    CargoCheckSpec {
        program: OsString::from("cargo"),
        args: vec![OsString::from("check")],
        current_dir: root.to_path_buf(),
        torch_cuda_environment: match backend {
            ResolvedBackend::Preconfigured => TorchCudaEnvironment::Inherit,
            ResolvedBackend::Cpu => TorchCudaEnvironment::Remove,
            ResolvedBackend::Cuda126 => TorchCudaEnvironment::Set(OsString::from("cu126")),
        },
    }
}

fn execute_cargo_check(spec: &CargoCheckSpec) -> Result<(), CliError> {
    let mut command = Command::new(&spec.program);
    command.args(&spec.args).current_dir(&spec.current_dir);
    match &spec.torch_cuda_environment {
        TorchCudaEnvironment::Inherit => {}
        TorchCudaEnvironment::Remove => {
            command.env_remove("TORCH_CUDA_VERSION");
        }
        TorchCudaEnvironment::Set(value) => {
            command.env("TORCH_CUDA_VERSION", value);
        }
    }
    let status = command
        .status()
        .map_err(|error| CliError::new(format!("could not start Cargo check: {error}")))?;
    if !status.success() {
        return Err(CliError::new(
            "Cargo check failed; fix the reported error and rerun rusttorch setup",
        ));
    }
    Ok(())
}

fn run() -> Result<(), CliError> {
    match parse_args(env::args_os().skip(1))? {
        CliAction::Help => println!("{HELP}"),
        CliAction::Version => println!("rusttorch {}", env!("CARGO_PKG_VERSION")),
        CliAction::Setup(request) => {
            let platform = Platform::current();
            let configured =
                is_setup_preconfigured(platform, |name| env::var_os(name).is_some(), Path::exists);
            let backend = resolve_setup_backend(request, platform, configured, detect_driver)?;
            let project_root = env::current_dir().map_err(|error| {
                CliError::new(format!("could not read current directory: {error}"))
            })?;
            ensure_project_root(&project_root)?;
            println!("Resolved backend: {backend}");
            if backend == ResolvedBackend::Preconfigured {
                println!("Using the existing LibTorch environment and Cargo configuration.");
            } else {
                let config_path = configure_project(&project_root, backend)?;
                println!("Project configuration: {}", config_path.display());
            }
            println!("Running Cargo check; the first LibTorch acquisition can be large.");
            execute_cargo_check(&cargo_check_spec(&project_root, backend))?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{ffi::OsString, fs};

    fn cargo_project() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("Cargo.toml"),
            "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        directory
    }

    fn write_config(root: &std::path::Path, contents: &str) {
        fs::create_dir(root.join(".cargo")).unwrap();
        fs::write(root.join(".cargo/config.toml"), contents).unwrap();
    }

    #[test]
    fn backend_parser_accepts_only_the_approved_names() {
        assert_eq!(parse_backend("auto").unwrap(), BackendRequest::Auto);
        assert_eq!(parse_backend("cpu").unwrap(), BackendRequest::Cpu);
        assert_eq!(parse_backend("cuda-12.6").unwrap(), BackendRequest::Cuda126);
        assert!(parse_backend("cuda").is_err());
        assert!(parse_backend("CUDA-12.6").is_err());
    }

    #[test]
    fn argument_parser_accepts_setup_help_and_version_only() {
        assert_eq!(
            parse_args(["setup", "--backend", "cpu"]).unwrap(),
            CliAction::Setup(BackendRequest::Cpu)
        );
        assert_eq!(parse_args(["--help"]).unwrap(), CliAction::Help);
        assert_eq!(parse_args(["--version"]).unwrap(), CliAction::Version);
        assert!(parse_args(["setup", "cpu"]).is_err());
        assert!(parse_args(["setup", "--backend", "cpu", "extra"]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn argument_parser_rejects_non_utf8_backend_names() {
        use std::os::unix::ffi::OsStringExt;

        assert!(
            parse_args([
                OsString::from("setup"),
                OsString::from("--backend"),
                OsString::from_vec(vec![0xff]),
            ])
            .is_err()
        );
    }

    #[test]
    fn platform_parser_recognizes_supported_operating_systems() {
        assert_eq!(Platform::from_os("linux"), Platform::Linux);
        assert_eq!(Platform::from_os("windows"), Platform::Windows);
        assert_eq!(Platform::from_os("macos"), Platform::Macos);
        assert_eq!(Platform::from_os("freebsd"), Platform::Other);
    }

    #[test]
    fn preconfiguration_detects_every_supported_environment_variable() {
        for configured_name in [
            "LIBTORCH_USE_PYTORCH",
            "LIBTORCH",
            "LIBTORCH_INCLUDE",
            "LIBTORCH_LIB",
        ] {
            assert!(is_libtorch_preconfigured(|name| name == configured_name));
        }
        assert!(!is_libtorch_preconfigured(|_| false));
    }

    #[test]
    fn setup_preconfiguration_includes_only_the_upstream_linux_system_path() {
        let libtorch = std::path::Path::new("/usr/lib/libtorch.so");

        assert!(is_setup_preconfigured(
            Platform::Linux,
            |_| false,
            |path| path == libtorch
        ));
        assert!(!is_setup_preconfigured(
            Platform::Macos,
            |_| false,
            |path| path == libtorch
        ));
        assert!(!is_setup_preconfigured(
            Platform::Linux,
            |_| false,
            |_| false
        ));
    }

    #[test]
    fn backend_resolution_probes_driver_only_when_the_platform_and_request_need_it() {
        use std::cell::Cell;

        for (request, platform, configured, expected_calls) in [
            (BackendRequest::Cpu, Platform::Linux, false, 0),
            (BackendRequest::Auto, Platform::Linux, true, 0),
            (BackendRequest::Auto, Platform::Macos, false, 0),
            (BackendRequest::Cuda126, Platform::Other, false, 0),
            (BackendRequest::Auto, Platform::Linux, false, 1),
            (BackendRequest::Cuda126, Platform::Windows, false, 1),
        ] {
            let calls = Cell::new(0);
            let _ = resolve_setup_backend(request, platform, configured, || {
                calls.set(calls.get() + 1);
                None
            });
            assert_eq!(calls.get(), expected_calls, "{request:?} on {platform:?}");
        }
    }

    #[test]
    fn driver_parser_uses_the_first_output_line() {
        assert_eq!(
            parse_driver_output("525.60.13\n535.54.03\n").unwrap(),
            DriverVersion::new(525, 60, 13)
        );
        assert_eq!(
            parse_driver_output("528.33\n").unwrap(),
            DriverVersion::new(528, 33, 0)
        );
        assert!(parse_driver_output("not-a-version\n").is_err());
    }

    #[test]
    fn auto_reuses_preconfiguration_and_selects_supported_acceleration() {
        assert_eq!(
            resolve_backend(BackendRequest::Auto, Platform::Linux, true, None).unwrap(),
            ResolvedBackend::Preconfigured
        );
        assert_eq!(
            resolve_backend(BackendRequest::Auto, Platform::Macos, false, None).unwrap(),
            ResolvedBackend::Cpu
        );
        assert_eq!(
            resolve_backend(
                BackendRequest::Auto,
                Platform::Linux,
                false,
                Some(DriverVersion::new(525, 60, 13)),
            )
            .unwrap(),
            ResolvedBackend::Cuda126
        );
        assert_eq!(
            resolve_backend(
                BackendRequest::Auto,
                Platform::Windows,
                false,
                Some(DriverVersion::new(528, 33, 0)),
            )
            .unwrap(),
            ResolvedBackend::Cuda126
        );
    }

    #[test]
    fn explicit_backend_refuses_active_preconfiguration() {
        assert!(resolve_backend(BackendRequest::Cpu, Platform::Linux, true, None).is_err());
        assert!(
            resolve_backend(
                BackendRequest::Cuda126,
                Platform::Linux,
                true,
                Some(DriverVersion::new(525, 60, 13)),
            )
            .is_err()
        );
    }

    #[test]
    fn explicit_cuda_requires_a_supported_platform_and_driver() {
        assert!(resolve_backend(BackendRequest::Cuda126, Platform::Macos, false, None).is_err());
        assert!(
            resolve_backend(
                BackendRequest::Cuda126,
                Platform::Linux,
                false,
                Some(DriverVersion::new(525, 59, 0)),
            )
            .is_err()
        );
        assert!(
            resolve_backend(
                BackendRequest::Cuda126,
                Platform::Windows,
                false,
                Some(DriverVersion::new(528, 32, 0)),
            )
            .is_err()
        );
    }

    #[test]
    fn auto_falls_back_to_cpu_for_incompatible_drivers() {
        assert_eq!(
            resolve_backend(
                BackendRequest::Auto,
                Platform::Linux,
                false,
                Some(DriverVersion::new(525, 59, 0)),
            )
            .unwrap(),
            ResolvedBackend::Cpu
        );
        assert_eq!(
            resolve_backend(
                BackendRequest::Auto,
                Platform::Windows,
                false,
                Some(DriverVersion::new(528, 32, 0)),
            )
            .unwrap(),
            ResolvedBackend::Cpu
        );
    }

    #[test]
    fn configuration_preserves_unrelated_toml_and_writes_cuda_selection() {
        let project = cargo_project();
        write_config(project.path(), "[alias]\nfast = \"check\"\n");

        let path = configure_project(project.path(), ResolvedBackend::Cuda126).unwrap();
        let document = fs::read_to_string(&path).unwrap();

        assert!(document.contains("fast = \"check\""));
        assert!(document.contains("target-dir = \"target/rusttorch/cuda-12.6\""));
        assert!(document.contains("RUSTTORCH_BACKEND = \"cuda-12.6\""));
        assert!(document.contains("TORCH_CUDA_VERSION = \"cu126\""));
    }

    #[test]
    fn configuration_switches_owned_cuda_selection_to_cpu() {
        let project = cargo_project();
        write_config(
            project.path(),
            "[build]\ntarget-dir = \"target/rusttorch/cuda-12.6\"\n\
             [env]\nRUSTTORCH_BACKEND = \"cuda-12.6\"\n\
             TORCH_CUDA_VERSION = \"cu126\"\nKEEP = \"yes\"\n",
        );

        let path = configure_project(project.path(), ResolvedBackend::Cpu).unwrap();
        let document = fs::read_to_string(path).unwrap();

        assert!(document.contains("target-dir = \"target/rusttorch/cpu\""));
        assert!(document.contains("RUSTTORCH_BACKEND = \"cpu\""));
        assert!(!document.contains("TORCH_CUDA_VERSION"));
        assert!(document.contains("KEEP = \"yes\""));
    }

    #[test]
    fn configuration_rejects_missing_manifest_before_writing() {
        let project = tempfile::tempdir().unwrap();

        let error = configure_project(project.path(), ResolvedBackend::Cpu).unwrap_err();

        assert!(error.to_string().contains("Cargo.toml"));
        assert!(!project.path().join(".cargo").exists());
    }

    #[test]
    fn configuration_preserves_invalid_toml_on_parse_failure() {
        let project = cargo_project();
        let original = b"[build\ntarget-dir = \"broken\"\n";
        write_config(project.path(), std::str::from_utf8(original).unwrap());

        assert!(configure_project(project.path(), ResolvedBackend::Cpu).is_err());

        assert_eq!(
            fs::read(project.path().join(".cargo/config.toml")).unwrap(),
            original
        );
    }

    #[test]
    fn configuration_refuses_an_unowned_target_directory() {
        let project = cargo_project();
        let original = "[build]\ntarget-dir = \"target/custom\"\n";
        write_config(project.path(), original);

        let error = configure_project(project.path(), ResolvedBackend::Cpu).unwrap_err();

        assert!(error.to_string().contains("build.target-dir"));
        assert_eq!(
            fs::read_to_string(project.path().join(".cargo/config.toml")).unwrap(),
            original
        );
    }

    #[test]
    fn configuration_does_not_adopt_an_unowned_same_value_marker() {
        let project = cargo_project();
        let original = "[env]\nRUSTTORCH_BACKEND = \"cuda-12.6\"\n";
        write_config(project.path(), original);

        let error = configure_project(project.path(), ResolvedBackend::Cuda126).unwrap_err();

        assert!(error.to_string().contains("RUSTTORCH_BACKEND"));
        assert_eq!(
            fs::read_to_string(project.path().join(".cargo/config.toml")).unwrap(),
            original
        );
    }

    #[test]
    fn configuration_refuses_an_unowned_cuda_environment_key() {
        let project = cargo_project();
        let original = "[env]\nTORCH_CUDA_VERSION = \"cu126\"\n";
        write_config(project.path(), original);

        let error = configure_project(project.path(), ResolvedBackend::Cpu).unwrap_err();

        assert!(error.to_string().contains("TORCH_CUDA_VERSION"));
        assert_eq!(
            fs::read_to_string(project.path().join(".cargo/config.toml")).unwrap(),
            original
        );
    }

    #[test]
    fn configuration_does_not_adopt_an_unowned_same_value_cuda_key() {
        let project = cargo_project();
        let original = "[env]\nTORCH_CUDA_VERSION = \"cu126\"\n";
        write_config(project.path(), original);

        let error = configure_project(project.path(), ResolvedBackend::Cuda126).unwrap_err();

        assert!(error.to_string().contains("TORCH_CUDA_VERSION"));
        assert_eq!(
            fs::read_to_string(project.path().join(".cargo/config.toml")).unwrap(),
            original
        );
    }

    #[test]
    fn configuration_preconfigured_leaves_existing_config_untouched() {
        let project = cargo_project();
        let original = "[build\nthis need not parse\n";
        write_config(project.path(), original);

        let path = configure_project(project.path(), ResolvedBackend::Preconfigured).unwrap();

        assert_eq!(
            fs::read_to_string(path).unwrap(),
            original,
            "preconfigured setup must not parse or rewrite user configuration"
        );
    }

    #[test]
    fn cargo_check_specs_describe_backend_environment_without_running_cargo() {
        let project = cargo_project();

        assert_eq!(
            cargo_check_spec(project.path(), ResolvedBackend::Cpu),
            CargoCheckSpec {
                program: OsString::from("cargo"),
                args: vec![OsString::from("check")],
                current_dir: project.path().to_path_buf(),
                torch_cuda_environment: TorchCudaEnvironment::Remove,
            }
        );
        assert_eq!(
            cargo_check_spec(project.path(), ResolvedBackend::Cuda126).torch_cuda_environment,
            TorchCudaEnvironment::Set(OsString::from("cu126"))
        );
        assert_eq!(
            cargo_check_spec(project.path(), ResolvedBackend::Preconfigured).torch_cuda_environment,
            TorchCudaEnvironment::Inherit
        );
    }

    #[test]
    fn unsuccessful_cargo_check_is_reported_without_network_access() {
        let project = cargo_project();
        let spec = CargoCheckSpec {
            program: OsString::from("rustc"),
            args: vec![OsString::from("--definitely-invalid-rusttorch-option")],
            current_dir: project.path().to_path_buf(),
            torch_cuda_environment: TorchCudaEnvironment::Inherit,
        };

        let error = execute_cargo_check(&spec).unwrap_err();

        assert!(error.to_string().contains("Cargo check failed"));
    }
}
