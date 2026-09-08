//! ggml's cmake enables OpenMP when it finds the headers, but llama-cpp-sys-2
//! never emits a link directive for it, so the final link fails on missing
//! ___kmpc_* symbols. Supply the flag here.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let prefix = std::process::Command::new("brew")
        .args(["--prefix", "libomp"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string());

    let Some(prefix) = prefix else {
        println!("cargo:warning=libomp not found; run `brew install libomp` if linking fails");
        return;
    };

    println!("cargo:rustc-link-search=native={prefix}/lib");
    println!("cargo:rustc-link-lib=static=omp");
}
