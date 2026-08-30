use std::{
    env,
    ffi::OsString,
    fmt,
    process::{self, Command},
};

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
                "explicit backends conflict with active LibTorch environment variables",
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

fn run() -> Result<(), CliError> {
    match parse_args(env::args_os().skip(1))? {
        CliAction::Help => println!("{HELP}"),
        CliAction::Version => println!("rusttorch {}", env!("CARGO_PKG_VERSION")),
        CliAction::Setup(request) => {
            let configured = is_libtorch_preconfigured(|name| env::var_os(name).is_some());
            let backend =
                resolve_backend(request, Platform::current(), configured, detect_driver())?;
            println!("Resolved backend: {backend}");
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
    use std::ffi::OsString;

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
}
