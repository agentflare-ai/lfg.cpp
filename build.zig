const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});

    const target = b.standardTargetOptions(.{});

    const enable_metal_opt = b.option(bool, "metal", "Enable Metal backend");
    const embed_metal_opt = b.option(bool, "metal_embed", "Embed Metal shader source");
    const enable_accelerate_opt = b.option(bool, "accelerate", "Enable Accelerate framework");
    const enable_openmp = b.option(bool, "openmp", "Enable OpenMP") orelse false;
    const use_native_opt = b.option(bool, "native", "Enable -march=native");
    const cpu_only = b.option(bool, "cpu_only", "Disable loading non-CPU backends") orelse false;

    const sched_max_copies = b.option(u32, "sched_max_copies", "GGML scheduler max copies") orelse 4;
    const sched_max_backends = b.option(u32, "sched_max_backends", "GGML scheduler max backends") orelse 16;

    // SPDLOG log levels: 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=CRITICAL, 6=OFF
    const spdlog_level = b.option(u32, "spdlog_level", "SPDLOG_ACTIVE_LEVEL (0-6, default 4=ERROR)") orelse 4;

    const build_demo = b.option(bool, "demo", "Build ImGui demo application") orelse false;

    const spdlog_dep = b.dependency("spdlog", .{});
    const spdlog_include = spdlog_dep.path("include");

    const build_baseline_combined =
        b.option(bool, "baseline_combined", "Build baseline combined library (no ISA suffix)") orelse false;
    const build_isa_combined =
        b.option(bool, "isa_combined", "Build ISA-optimized combined libraries") orelse true;
    const build_all_targets =
        b.option(bool, "all_targets", "Build combined libraries for common targets") orelse true;
    const windows_shared_opt =
        b.option(bool, "windows_shared", "Build Windows shared libraries (DLLs)");
    const windows_shared = windows_shared_opt orelse hasMingwLibs(b);
    const windows_arm64 =
        b.option(bool, "windows_arm64", "Build Windows ARM64 libraries") orelse true;

    if (build_all_targets) {
        var target_queries = std.ArrayList(std.Target.Query).empty;
        defer target_queries.deinit(b.allocator);

        target_queries.append(b.allocator, .{ .cpu_arch = .aarch64, .os_tag = .macos }) catch @panic("oom");
        target_queries.append(b.allocator, .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu }) catch @panic("oom");
        target_queries.append(b.allocator, .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu }) catch @panic("oom");
        target_queries.append(b.allocator, .{ .cpu_arch = .x86_64, .os_tag = .windows }) catch @panic("oom");
        if (windows_arm64) {
            target_queries.append(b.allocator, .{ .cpu_arch = .aarch64, .os_tag = .windows }) catch @panic("oom");
        }

        for (target_queries.items) |query| {
            const resolved = b.resolveTargetQuery(query);
            addTargetArtifacts(
                b,
                resolved,
                optimize,
                enable_metal_opt,
                embed_metal_opt,
                enable_accelerate_opt,
                enable_openmp,
                use_native_opt,
                sched_max_copies,
                sched_max_backends,
                spdlog_include,
                spdlog_level,
                cpu_only,
                build_baseline_combined,
                build_isa_combined,
                windows_shared,
                false,
                true,
                false,
            );
        }
    } else {
        addTargetArtifacts(
            b,
            target,
            optimize,
            enable_metal_opt,
            embed_metal_opt,
            enable_accelerate_opt,
            enable_openmp,
            use_native_opt,
            sched_max_copies,
            sched_max_backends,
            spdlog_include,
            spdlog_level,
            cpu_only,
            build_baseline_combined,
            build_isa_combined,
            windows_shared,
            true,
            false,
            build_demo,
        );
    }
}

fn addTargetArtifacts(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    enable_metal_opt: ?bool,
    embed_metal_opt: ?bool,
    enable_accelerate_opt: ?bool,
    enable_openmp: bool,
    use_native_opt: ?bool,
    sched_max_copies: u32,
    sched_max_backends: u32,
    spdlog_include: std.Build.LazyPath,
    spdlog_level: u32,
    cpu_only: bool,
    build_baseline_combined: bool,
    build_isa_combined: bool,
    windows_shared: bool,
    build_tools: bool,
    multi_target: bool,
    build_demo: bool,
) void {
    var effective_target = target;
    if (target.result.os.tag == .macos and target.result.cpu.arch == .aarch64) {
        effective_target = resolveIsaTarget(
            b,
            target,
            &std.Target.aarch64.cpu.apple_m1,
            std.Target.Cpu.Feature.Set.empty,
        );
    }

    const is_darwin = effective_target.result.os.tag == .macos;
    const enable_metal = enable_metal_opt orelse is_darwin;
    const enable_accelerate = enable_accelerate_opt orelse is_darwin;
    const embed_metal = embed_metal_opt orelse enable_metal;
    const use_native = (use_native_opt orelse effective_target.query.isNative()) and effective_target.query.isNative();

    const sysroot = if (effective_target.result.os.tag == .macos) getMacosSysroot(b) else null;
    const framework_path = if (sysroot) |path| b.fmt("{s}/System/Library/Frameworks", .{path}) else null;
    const private_framework_path = if (sysroot) |path| b.fmt("{s}/System/Library/PrivateFrameworks", .{path}) else null;
    const base_flags = makeFlags(b, optimize, use_native, enable_openmp, extraFlagsForTarget(b, effective_target), sysroot);
    const c_flags_slice = base_flags.c;
    const cxx_flags_slice = base_flags.cxx;

    const ggml = addGgmlLibrary(
        b,
        effective_target,
        optimize,
        enable_metal,
        embed_metal,
        enable_accelerate,
        enable_openmp,
        sched_max_copies,
        sched_max_backends,
        c_flags_slice,
        cxx_flags_slice,
        framework_path,
        private_framework_path,
        cpu_only,
    );
    const llama_core = addLlamaCore(b, effective_target, optimize, ggml, cxx_flags_slice, framework_path, private_framework_path);
    const lfg_vision = addLiquidVision(b, effective_target, optimize, ggml, spdlog_include, spdlog_level, cxx_flags_slice, framework_path, private_framework_path);
    const lfg_core = addLiquidCore(b, effective_target, optimize, ggml, lfg_vision, spdlog_include, spdlog_level, cxx_flags_slice, framework_path, private_framework_path);

    if (build_tools) {
        addExecutables(b, effective_target, optimize, ggml, llama_core, lfg_vision, lfg_core, spdlog_include, cxx_flags_slice, c_flags_slice, framework_path, private_framework_path, sysroot, build_demo);
        b.installArtifact(ggml);
        b.installArtifact(llama_core);
        b.installArtifact(lfg_vision);
        b.installArtifact(lfg_core);
    }

    const shared_link_args = makeSharedLinkArgs(b, effective_target, enable_metal, enable_accelerate, enable_openmp, sysroot);
    const target_label = targetLabel(b, effective_target);

    if (build_baseline_combined) {
        const suffix = if (multi_target) b.fmt("-{s}", .{target_label}) else "";
        addCombinedLibrary(b, ggml, lfg_core, lfg_vision, effective_target, optimize, suffix, shared_link_args, windows_shared);
    }

    if (build_isa_combined) {
        addIsaCombinedLibraries(
            b,
            effective_target,
            optimize,
            enable_metal,
            embed_metal,
            enable_accelerate,
            enable_openmp,
            sched_max_copies,
            sched_max_backends,
            spdlog_include,
            spdlog_level,
            shared_link_args,
            target_label,
            sysroot,
            framework_path,
            private_framework_path,
            windows_shared,
            cpu_only,
        );
    }
}

fn addCombinedLibrary(
    b: *std.Build,
    ggml: *std.Build.Step.Compile,
    lfg_core: *std.Build.Step.Compile,
    lfg_vision: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    name_suffix: []const u8,
    shared_link_args: []const []const u8,
    windows_shared: bool,
) void {
    const host_os = b.graph.host.result.os.tag;
    if (host_os == .windows) {
        return;
    }
    const strip_symbols = optimize == .ReleaseFast;
    const is_windows = target.result.os.tag == .windows;
    const build_shared = if (is_windows) windows_shared else true;
    const shared_ext = if (target.result.os.tag == .macos) "dylib" else if (is_windows) "dll" else "so";
    const static_ext = if (is_windows) "lib" else "a";
    const static_name = if (name_suffix.len == 0)
        b.fmt("liblfg.{s}", .{static_ext})
    else
        b.fmt("liblfg{s}.{s}", .{ name_suffix, static_ext });
    const static_windows_name = if (is_windows)
        (if (name_suffix.len == 0)
            "liblfg-static.lib"
        else
            b.fmt("liblfg{s}-static.lib", .{name_suffix}))
    else
        static_name;
    const shared_name = if (name_suffix.len == 0)
        b.fmt("liblfg.{s}", .{shared_ext})
    else
        b.fmt("liblfg{s}.{s}", .{ name_suffix, shared_ext });
    const implib_name = if (is_windows)
        (if (name_suffix.len == 0)
            "liblfg.lib"
        else
            b.fmt("liblfg{s}.lib", .{name_suffix}))
    else
        "";

    const target_triple = target.result.zigTriple(b.allocator) catch @panic("oom");
    const script =
        "set -e\n" ++
        "out_static=\"$1\"; out_shared=\"$2\"; out_implib=\"$3\"; core=\"$4\"; ggml=\"$5\"; vision=\"$6\"; zig=\"$7\"; root=\"$8\"; triple=\"$9\"; mode=\"${10}\"; strip=\"${11}\"\n" ++
        "if [ \"$out_shared\" = \"__no_shared__\" ]; then out_shared=\"\"; fi\n" ++
        "if [ \"$out_implib\" = \"__no_implib__\" ]; then out_implib=\"\"; fi\n" ++
        "shift 11\n" ++
        "case \"$core\" in /*) ;; *) core=\"$root/$core\";; esac\n" ++
        "case \"$ggml\" in /*) ;; *) ggml=\"$root/$ggml\";; esac\n" ++
        "case \"$vision\" in /*) ;; *) vision=\"$root/$vision\";; esac\n" ++
        "tmp=$(mktemp -d)\n" ++
        "cleanup(){ rm -rf \"$tmp\"; }\n" ++
        "trap cleanup EXIT\n" ++
        "extract_archive() {\n" ++
        "  archive=\"$1\"; prefix=\"$2\"\n" ++
        "  \"$zig\" ar t \"$archive\" | awk '/\\.o$/ {count[$0]++; printf \"%d\\t%s\\n\", count[$0], $0}' > \"$tmp/${prefix}_members.txt\"\n" ++
        "  while IFS=$'\\t' read -r idx name; do\n" ++
        "    [ -z \"$name\" ] && continue\n" ++
        "    \"$zig\" ar xN \"$idx\" \"$archive\" \"$name\" >/dev/null 2>&1\n" ++
        "    base=$(basename \"$name\")\n" ++
        "    chmod u+rw \"$base\" || true\n" ++
        "    mv \"$base\" \"$tmp/${prefix}__${idx}__${base}\"\n" ++
        "  done < \"$tmp/${prefix}_members.txt\"\n" ++
        "}\n" ++
        "cd \"$tmp\"\n" ++
        "extract_archive \"$core\" core\n" ++
        "extract_archive \"$ggml\" ggml\n" ++
        "extract_archive \"$vision\" vision\n" ++
        "find \"$tmp\" -maxdepth 1 -type f -name '*.o' -print > \"$tmp/objects.txt\"\n" ++
        "\"$zig\" ar rcs \"$out_static\" @\"$tmp/objects.txt\"\n" ++
        "if [ -n \"$out_shared\" ]; then\n" ++
        "  strip_flag=\"\"\n" ++
        "  if [ \"$strip\" = \"1\" ]; then strip_flag=\"-s\"; fi\n" ++
        "  implib_flag=\"\"\n" ++
        "  if [ -n \"$out_implib\" ]; then implib_flag=\"-Wl,--out-implib,$out_implib\"; fi\n" ++
        "  if [ \"$mode\" = \"macos\" ]; then\n" ++
        "    \"$zig\" cc -target \"$triple\" -dynamiclib $strip_flag -o \"$out_shared\" @\"$tmp/objects.txt\" $implib_flag \"$@\"\n" ++
        "  else\n" ++
        "    \"$zig\" cc -target \"$triple\" -shared $strip_flag -o \"$out_shared\" @\"$tmp/objects.txt\" $implib_flag \"$@\"\n" ++
        "  fi\n" ++
        "fi\n";

    const pack = b.addSystemCommand(&.{ "sh", "-c", script, "pack" });
    const out_static = pack.addOutputFileArg(static_windows_name);
    const out_shared = if (build_shared) pack.addOutputFileArg(shared_name) else null;
    const out_implib = if (is_windows and build_shared) pack.addOutputFileArg(implib_name) else null;
    if (!build_shared) {
        pack.addArg("__no_shared__");
    }
    if (!is_windows or !build_shared) {
        pack.addArg("__no_implib__");
    }
    pack.addFileArg(lfg_core.getEmittedBin());
    pack.addFileArg(ggml.getEmittedBin());
    pack.addFileArg(lfg_vision.getEmittedBin());
    pack.addArg(b.graph.zig_exe);
    pack.addArg(b.pathFromRoot("."));
    pack.addArg(target_triple);
    pack.addArg(if (target.result.os.tag == .macos) "macos" else "unix");
    pack.addArg(if (strip_symbols) "1" else "0");
    if (shared_link_args.len > 0) {
        pack.addArgs(shared_link_args);
    }

    const install_static = b.addInstallLibFile(out_static, static_windows_name);
    install_static.step.dependOn(&pack.step);
    b.getInstallStep().dependOn(&install_static.step);

    if (build_shared) {
        const install_shared = b.addInstallLibFile(out_shared.?, shared_name);
        install_shared.step.dependOn(&pack.step);
        b.getInstallStep().dependOn(&install_shared.step);
    }

    if (is_windows and build_shared) {
        const install_implib = b.addInstallLibFile(out_implib.?, implib_name);
        install_implib.step.dependOn(&pack.step);
        b.getInstallStep().dependOn(&install_implib.step);
    }
}

fn createModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    pic: ?bool,
    strip: ?bool,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
) *std.Build.Module {
    const module = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        .pic = pic,
        .strip = strip,
    });
    if (framework_path) |path| {
        module.addFrameworkPath(.{ .cwd_relative = path });
    }
    if (private_framework_path) |path| {
        module.addFrameworkPath(.{ .cwd_relative = path });
    }
    return module;
}

fn addCommonDefines(
    b: *std.Build,
    step: *std.Build.Step.Compile,
    is_darwin: bool,
    is_linux: bool,
    enable_metal: bool,
    enable_accelerate: bool,
    enable_openmp: bool,
    sched_max_copies: u32,
    sched_max_backends: u32,
) void {
    step.root_module.addCMacro("GGML_VERSION", "\"1.0\"");
    step.root_module.addCMacro("GGML_COMMIT", "\"unknown\"");
    step.root_module.addCMacro("GGML_SCHED_MAX_COPIES", b.fmt("{d}", .{sched_max_copies}));
    step.root_module.addCMacro("GGML_SCHED_MAX_BACKENDS", b.fmt("{d}", .{sched_max_backends}));
    step.root_module.addCMacro("_XOPEN_SOURCE", "600");
    step.root_module.addCMacro("GGML_USE_CPU", "1");
    if (is_darwin) {
        step.root_module.addCMacro("_DARWIN_C_SOURCE", "1");
    }
    if (is_linux) {
        step.root_module.addCMacro("_GNU_SOURCE", "1");
    }

    if (enable_metal) {
        step.root_module.addCMacro("GGML_USE_METAL", "1");
    }
    if (enable_accelerate) {
        step.root_module.addCMacro("GGML_USE_ACCELERATE", "1");
        step.root_module.addCMacro("ACCELERATE_NEW_LAPACK", "1");
        step.root_module.addCMacro("ACCELERATE_LAPACK_ILP64", "1");
    }
    if (enable_openmp) {
        step.root_module.addCMacro("GGML_USE_OPENMP", "1");
    }
}

fn addGgmlLibrary(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    enable_metal: bool,
    embed_metal: bool,
    enable_accelerate: bool,
    enable_openmp: bool,
    sched_max_copies: u32,
    sched_max_backends: u32,
    c_flags: []const []const u8,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    cpu_only: bool,
) *std.Build.Step.Compile {
    const ggml_mod = createModule(b, target, optimize, true, optimize == .ReleaseFast, framework_path, private_framework_path);
    const ggml = b.addLibrary(.{
        .name = "ggml",
        .root_module = ggml_mod,
        .linkage = .static,
    });

    addCommonDefines(
        b,
        ggml,
        target.result.os.tag == .macos,
        target.result.os.tag == .linux,
        enable_metal,
        enable_accelerate,
        enable_openmp,
        sched_max_copies,
        sched_max_backends,
    );
    if (cpu_only) {
        ggml.root_module.addCMacro("GGML_BACKEND_CPU_ONLY", "1");
    }

    ggml.addIncludePath(b.path("src/ggml"));
    ggml.addIncludePath(b.path("src/ggml/ggml-cpu"));

    const ggml_src = "src/ggml";
    const ggml_cpu = "src/ggml/ggml-cpu";
    const ggml_metal = "src/ggml/ggml-metal";

    const ggml_c = &[_][]const u8{
        ggml_src ++ "/ggml.c",
        ggml_src ++ "/ggml-alloc.c",
        ggml_src ++ "/ggml-quants.c",
    };
    const ggml_cpp = &[_][]const u8{
        ggml_src ++ "/ggml.cpp",
        ggml_src ++ "/ggml-backend.cpp",
        ggml_src ++ "/ggml-backend-reg.cpp",
        ggml_src ++ "/ggml-opt.cpp",
        ggml_src ++ "/ggml-threading.cpp",
        ggml_src ++ "/gguf.cpp",
    };

    ggml.addCSourceFiles(.{ .files = ggml_c, .flags = c_flags });
    ggml.addCSourceFiles(.{ .files = ggml_cpp, .flags = cxx_flags });

    const cpu_c = &[_][]const u8{
        ggml_cpu ++ "/ggml-cpu.c",
        ggml_cpu ++ "/quants.c",
    };
    const cpu_cpp = &[_][]const u8{
        ggml_cpu ++ "/ggml-cpu.cpp",
        ggml_cpu ++ "/repack.cpp",
        ggml_cpu ++ "/hbm.cpp",
        ggml_cpu ++ "/traits.cpp",
        ggml_cpu ++ "/binary-ops.cpp",
        ggml_cpu ++ "/unary-ops.cpp",
        ggml_cpu ++ "/vec.cpp",
        ggml_cpu ++ "/ops.cpp",
        ggml_cpu ++ "/amx/amx.cpp",
        ggml_cpu ++ "/amx/mmq.cpp",
    };

    ggml.addCSourceFiles(.{ .files = cpu_c, .flags = c_flags });
    ggml.addCSourceFiles(.{ .files = cpu_cpp, .flags = cxx_flags });

    if (target.result.cpu.arch == .aarch64 or target.result.cpu.arch == .arm) {
        const arm_c = &[_][]const u8{ggml_cpu ++ "/arch/arm/quants.c"};
        const arm_cpp = &[_][]const u8{ggml_cpu ++ "/arch/arm/repack.cpp"};
        ggml.addCSourceFiles(.{ .files = arm_c, .flags = c_flags });
        ggml.addCSourceFiles(.{ .files = arm_cpp, .flags = cxx_flags });
    }

    if (target.result.cpu.arch == .x86_64 or target.result.cpu.arch == .x86) {
        const x86_c = &[_][]const u8{ggml_cpu ++ "/arch/x86/quants.c"};
        const x86_cpp = &[_][]const u8{ggml_cpu ++ "/arch/x86/repack.cpp"};
        ggml.addCSourceFiles(.{ .files = x86_c, .flags = c_flags });
        ggml.addCSourceFiles(.{ .files = x86_cpp, .flags = cxx_flags });
    }

    if (enable_metal and target.result.os.tag == .macos) {
        if (framework_path) |path| {
            ggml.root_module.addFrameworkPath(.{ .cwd_relative = path });
        }
        if (private_framework_path) |path| {
            ggml.root_module.addFrameworkPath(.{ .cwd_relative = path });
        }
        ggml.addIncludePath(b.path(ggml_metal));
        ggml.root_module.addCMacro("GGML_METAL", "1");

        if (embed_metal) {
            ggml.root_module.addCMacro("GGML_METAL_EMBED_LIBRARY", "1");
            const gen = b.addSystemCommand(&.{ "python3", "scripts/gen_metal_embed.py" });
            gen.addArg("--common");
            gen.addFileArg(b.path(ggml_src ++ "/ggml-common.h"));
            gen.addArg("--metal");
            gen.addFileArg(b.path(ggml_metal ++ "/ggml-metal.metal"));
            gen.addArg("--impl");
            gen.addFileArg(b.path(ggml_metal ++ "/ggml-metal-impl.h"));
            gen.addArg("--out-metal");
            const out_metal = gen.addOutputFileArg("ggml-metal-embed.metal");
            gen.addArg("--out-asm");
            const out_asm = gen.addOutputFileArg("ggml-metal-embed.s");
            ggml.addAssemblyFile(out_asm);
            ggml.step.dependOn(&gen.step);
            _ = out_metal;
        }

        const metal_cpp = &[_][]const u8{
            ggml_metal ++ "/ggml-metal.cpp",
            ggml_metal ++ "/ggml-metal-device.cpp",
            ggml_metal ++ "/ggml-metal-common.cpp",
            ggml_metal ++ "/ggml-metal-ops.cpp",
        };
        const metal_objc = &[_][]const u8{
            ggml_metal ++ "/ggml-metal-device.m",
            ggml_metal ++ "/ggml-metal-context.m",
        };
        ggml.addCSourceFiles(.{ .files = metal_cpp, .flags = cxx_flags });
        ggml.addCSourceFiles(.{ .files = metal_objc, .flags = c_flags });

        ggml.linkFramework("Foundation");
        ggml.linkFramework("Metal");
        ggml.linkFramework("MetalKit");
    }

    if (enable_accelerate and target.result.os.tag == .macos) {
        if (framework_path) |path| {
            ggml.root_module.addFrameworkPath(.{ .cwd_relative = path });
        }
        if (private_framework_path) |path| {
            ggml.root_module.addFrameworkPath(.{ .cwd_relative = path });
        }
        ggml.linkFramework("Accelerate");
    }

    if (enable_openmp) {
        ggml.linkSystemLibrary("omp");
    }

    if (target.result.os.tag == .linux) {
        ggml.linkSystemLibrary("dl");
    }

    if (target.result.os.tag != .windows) {
        ggml.linkSystemLibrary("pthread");
    }

    return ggml;
}

fn addLlamaCore(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml: *std.Build.Step.Compile,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
) *std.Build.Step.Compile {
    const llama_mod = createModule(b, target, optimize, true, optimize == .ReleaseFast, framework_path, private_framework_path);
    const llama_core = b.addLibrary(.{
        .name = "llama_core_local",
        .root_module = llama_mod,
        .linkage = .static,
    });

    llama_core.addIncludePath(b.path("third_party/llama.cpp/src"));
    llama_core.addIncludePath(b.path("third_party/llama.cpp/include"));
    llama_core.addIncludePath(b.path("src/ggml"));

    const src_files = collectFiles(b.allocator, "third_party/llama.cpp/src", &[_][]const u8{".cpp"}) catch @panic("oom");
    const model_files = collectFiles(b.allocator, "third_party/llama.cpp/src/models", &[_][]const u8{".cpp"}) catch @panic("oom");

    llama_core.addCSourceFiles(.{ .files = src_files, .flags = cxx_flags });
    llama_core.addCSourceFiles(.{ .files = model_files, .flags = cxx_flags });

    llama_core.linkLibrary(ggml);

    return llama_core;
}

fn addLiquidVision(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml: *std.Build.Step.Compile,
    spdlog_include: std.Build.LazyPath,
    spdlog_level: u32,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
) *std.Build.Step.Compile {
    const vision_mod = createModule(b, target, optimize, true, optimize == .ReleaseFast, framework_path, private_framework_path);
    const vision = b.addLibrary(.{
        .name = "lfg_vision",
        .root_module = vision_mod,
        .linkage = .static,
    });

    vision.addIncludePath(b.path("src/vision"));
    vision.addIncludePath(spdlog_include);
    vision.addIncludePath(b.path("src/ggml"));

    // Set spdlog compile-time log level
    vision.root_module.addCMacro("SPDLOG_ACTIVE_LEVEL", b.fmt("{d}", .{spdlog_level}));

    vision.addCSourceFiles(.{ .files = &[_][]const u8{ "src/vision/clip.cpp", "src/vision/siglip.cpp" }, .flags = cxx_flags });

    vision.linkLibrary(ggml);

    return vision;
}

fn addLiquidCore(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml: *std.Build.Step.Compile,
    vision: *std.Build.Step.Compile,
    spdlog_include: std.Build.LazyPath,
    spdlog_level: u32,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
) *std.Build.Step.Compile {
    const core_mod = createModule(b, target, optimize, true, optimize == .ReleaseFast, framework_path, private_framework_path);
    const core = b.addLibrary(.{
        .name = "lfg_core",
        .root_module = core_mod,
        .linkage = .static,
    });

    core.addIncludePath(b.path("src/inference"));
    core.addIncludePath(b.path("src/loader"));
    core.addIncludePath(b.path("src/vision"));
    core.addIncludePath(b.path("src/ggml"));
    core.addIncludePath(b.path("third_party/llama.cpp/src"));
    core.addIncludePath(b.path("third_party/llama.cpp/vendor"));
    core.addIncludePath(b.path("third_party/llama.cpp/common"));
    core.addIncludePath(spdlog_include);

    // Set spdlog compile-time log level
    core.root_module.addCMacro("SPDLOG_ACTIVE_LEVEL", b.fmt("{d}", .{spdlog_level}));

    const inf = collectFiles(b.allocator, "src/inference", &[_][]const u8{".cpp"}) catch @panic("oom");
    const inf_models = collectFiles(b.allocator, "src/inference/models", &[_][]const u8{".cpp"}) catch @panic("oom");
    const loader = collectFiles(b.allocator, "src/loader", &[_][]const u8{".cpp"}) catch @panic("oom");

    core.addCSourceFiles(.{ .files = inf, .flags = cxx_flags });
    core.addCSourceFiles(.{ .files = inf_models, .flags = cxx_flags });
    core.addCSourceFiles(.{ .files = loader, .flags = cxx_flags });
    core.addCSourceFiles(.{ .files = &[_][]const u8{
        "third_party/llama.cpp/common/peg-parser.cpp",
        "third_party/llama.cpp/common/unicode.cpp",
    }, .flags = cxx_flags });

    core.linkLibrary(ggml);
    core.linkLibrary(vision);

    if (target.result.os.tag == .linux) {
        core.linkSystemLibrary("dl");
    }
    if (target.result.os.tag != .windows) {
        core.linkSystemLibrary("pthread");
    }

    return core;
}

fn addExecutables(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml: *std.Build.Step.Compile,
    llama_core: *std.Build.Step.Compile,
    vision: *std.Build.Step.Compile,
    lfg_core: *std.Build.Step.Compile,
    spdlog_include: std.Build.LazyPath,
    cxx_flags: []const []const u8,
    c_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    sysroot: ?[]const u8,
    build_demo: bool,
) void {
    _ = vision;

    const exe = addExe(b, target, optimize, "lfg_cli", &[_][]const u8{"src/main.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    exe.linkLibrary(lfg_core);
    addCommonExeLinks(exe, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(exe);

    const eval = addExe(b, target, optimize, "lfg-eval", &[_][]const u8{"src/eval/eval.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    eval.addIncludePath(b.path("third_party/llama.cpp/vendor"));
    eval.addIncludePath(b.path("third_party/llama.cpp"));
    eval.addIncludePath(b.path("third_party/llama.cpp/include"));
    eval.addIncludePath(b.path("src/ggml"));
    eval.linkLibrary(lfg_core);
    eval.linkLibrary(ggml);
    addCommonExeLinks(eval, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(eval);

    const llama_compare = addExe(b, target, optimize, "llama-compare", &[_][]const u8{"src/eval/llama_compare.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    llama_compare.addIncludePath(b.path("third_party/llama.cpp/include"));
    llama_compare.addIncludePath(b.path("src/ggml"));
    llama_compare.linkLibrary(llama_core);
    addCommonExeLinks(llama_compare, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(llama_compare);

    const lfg_compare = addExe(b, target, optimize, "lfg-compare", &[_][]const u8{"src/eval/lfg_compare.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    lfg_compare.linkLibrary(lfg_core);
    lfg_compare.linkLibrary(ggml);
    addCommonExeLinks(lfg_compare, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(lfg_compare);

    const lfg_struct = addExe(b, target, optimize, "lfg-structured-compare", &[_][]const u8{"src/eval/lfg_structured_compare.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    lfg_struct.linkLibrary(lfg_core);
    lfg_struct.linkLibrary(ggml);
    addCommonExeLinks(lfg_struct, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(lfg_struct);

    const bare_thinking = addExe(b, target, optimize, "bare-thinking-test", &[_][]const u8{"src/eval/bare_thinking_test.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    bare_thinking.linkLibrary(lfg_core);
    bare_thinking.linkLibrary(ggml);
    addCommonExeLinks(bare_thinking, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(bare_thinking);

    const verify_thinking = addExe(b, target, optimize, "verify-thinking-structured", &[_][]const u8{"src/eval/verify_thinking_structured.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    verify_thinking.linkLibrary(lfg_core);
    verify_thinking.linkLibrary(ggml);
    addCommonExeLinks(verify_thinking, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(verify_thinking);

    const compare_rmsnorm = addExe(b, target, optimize, "compare-rmsnorm", &[_][]const u8{"src/eval/compare_rmsnorm.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    compare_rmsnorm.linkLibrary(lfg_core);
    compare_rmsnorm.linkLibrary(ggml);
    addCommonExeLinks(compare_rmsnorm, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(compare_rmsnorm);

    const bench_tool_ranking = addExe(b, target, optimize, "bench-tool-ranking", &[_][]const u8{"src/eval/bench_tool_ranking.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    bench_tool_ranking.linkLibrary(lfg_core);
    bench_tool_ranking.linkLibrary(ggml);
    addCommonExeLinks(bench_tool_ranking, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(bench_tool_ranking);

    const eval_entropy = addExe(b, target, optimize, "eval-entropy-retrieval", &[_][]const u8{"src/eval/eval_entropy_retrieval.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    eval_entropy.linkLibrary(lfg_core);
    eval_entropy.linkLibrary(ggml);
    addCommonExeLinks(eval_entropy, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(eval_entropy);

    const bench_confidence = addExe(b, target, optimize, "bench-confidence-overhead", &[_][]const u8{"src/eval/bench_confidence_overhead.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    bench_confidence.linkLibrary(lfg_core);
    bench_confidence.linkLibrary(ggml);
    addCommonExeLinks(bench_confidence, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(bench_confidence);

    const bench_surprise = addExe(b, target, optimize, "bench-surprise-overhead", &[_][]const u8{"src/eval/bench_surprise_overhead.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    bench_surprise.linkLibrary(lfg_core);
    bench_surprise.linkLibrary(ggml);
    addCommonExeLinks(bench_surprise, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(bench_surprise);

    const eval_tool_sim = addExe(b, target, optimize, "eval-tool-similarity", &[_][]const u8{"src/eval/eval_tool_similarity.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    eval_tool_sim.linkLibrary(lfg_core);
    eval_tool_sim.linkLibrary(ggml);
    addCommonExeLinks(eval_tool_sim, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(eval_tool_sim);

    const eval_bert_parity = addExe(b, target, optimize, "eval-bert-parity", &[_][]const u8{"src/eval/eval_bert_parity.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    eval_bert_parity.addIncludePath(b.path("third_party/llama.cpp/include"));
    eval_bert_parity.addIncludePath(b.path("src/ggml"));
    eval_bert_parity.linkLibrary(lfg_core);
    eval_bert_parity.linkLibrary(llama_core);
    eval_bert_parity.linkLibrary(ggml);
    addCommonExeLinks(eval_bert_parity, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(eval_bert_parity);

    const llama_struct = addExe(b, target, optimize, "llama-structured-compare", &[_][]const u8{
        "src/eval/llama_structured_compare.cpp",
        "src/inference/json_schema_to_grammar.cpp",
    }, spdlog_include, cxx_flags, framework_path, private_framework_path);
    llama_struct.addIncludePath(b.path("third_party/llama.cpp/include"));
    llama_struct.addIncludePath(b.path("src/ggml"));
    llama_struct.addIncludePath(spdlog_include);
    llama_struct.linkLibrary(llama_core);
    addCommonExeLinks(llama_struct, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(llama_struct);

    if (build_demo and target.result.os.tag == .macos) {
        const glfw_lib = addGlfwLibrary(b, target, optimize, c_flags, framework_path, private_framework_path, sysroot);
        const imgui_lib = addImguiLibrary(b, target, optimize, glfw_lib, cxx_flags, framework_path, private_framework_path);
        const demo = addExe(b, target, optimize, "lfg-demo", &[_][]const u8{"tools/demo/main.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
        demo.linkLibrary(lfg_core);
        demo.linkLibrary(imgui_lib);
        demo.linkLibrary(glfw_lib);
        demo.linkLibrary(ggml);
        demo.addIncludePath(b.path("third_party/imgui"));
        demo.addIncludePath(b.path("third_party/imgui/backends"));
        const glfw_dep = b.dependency("glfw", .{});
        demo.addIncludePath(glfw_dep.path("include"));
        // ObjC file dialog source
        var objc_flags = std.ArrayList([]const u8).empty;
        objc_flags.append(b.allocator, "-fno-objc-arc") catch @panic("oom");
        objc_flags.append(b.allocator, "-Wno-deprecated-declarations") catch @panic("oom");
        objc_flags.appendSlice(b.allocator, c_flags) catch @panic("oom");
        const objc_flags_slice = objc_flags.toOwnedSlice(b.allocator) catch @panic("oom");
        demo.addCSourceFiles(.{ .files = &[_][]const u8{"tools/demo/file_dialog.m"}, .flags = objc_flags_slice });
        demo.addIncludePath(b.path("tools/demo"));
        demo.linkFramework("AppKit");
        demo.linkFramework("UniformTypeIdentifiers");
        demo.linkFramework("OpenGL");
        demo.linkFramework("Cocoa");
        demo.linkFramework("IOKit");
        demo.linkFramework("CoreFoundation");
        demo.linkFramework("CoreVideo");
        addCommonExeLinks(demo, target, framework_path, private_framework_path, sysroot);
        b.installArtifact(demo);
    }

    addBenchmarks(b, target, optimize, ggml, llama_core, lfg_core, spdlog_include, cxx_flags, framework_path, private_framework_path, sysroot);
    addTests(b, target, optimize, ggml, lfg_core, spdlog_include, cxx_flags, framework_path, private_framework_path, sysroot);
}

fn addGlfwLibrary(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    c_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    sysroot: ?[]const u8,
) *std.Build.Step.Compile {
    const glfw_dep = b.dependency("glfw", .{});
    const glfw_mod = createModule(b, target, optimize, true, optimize == .ReleaseFast, framework_path, private_framework_path);
    const glfw = b.addLibrary(.{
        .name = "glfw",
        .root_module = glfw_mod,
        .linkage = .static,
    });

    glfw.addIncludePath(glfw_dep.path("include"));
    glfw.addIncludePath(glfw_dep.path("src"));

    // Common C sources
    const common_c = &[_][]const u8{
        "src/context.c",
        "src/init.c",
        "src/input.c",
        "src/monitor.c",
        "src/platform.c",
        "src/vulkan.c",
        "src/window.c",
        "src/egl_context.c",
        "src/osmesa_context.c",
        "src/null_init.c",
        "src/null_joystick.c",
        "src/null_monitor.c",
        "src/null_window.c",
    };

    // macOS C sources
    const macos_c = &[_][]const u8{
        "src/cocoa_time.c",
        "src/posix_module.c",
        "src/posix_thread.c",
        "src/posix_poll.c",
    };

    // macOS ObjC sources
    const macos_m = &[_][]const u8{
        "src/cocoa_init.m",
        "src/cocoa_joystick.m",
        "src/cocoa_monitor.m",
        "src/cocoa_window.m",
        "src/nsgl_context.m",
    };

    // Build C flags with _GLFW_COCOA define
    var glfw_c_flags = std.ArrayList([]const u8).empty;
    glfw_c_flags.append(b.allocator, "-D_GLFW_COCOA") catch @panic("oom");
    glfw_c_flags.append(b.allocator, "-Wno-deprecated-declarations") catch @panic("oom");
    glfw_c_flags.appendSlice(b.allocator, c_flags) catch @panic("oom");
    const glfw_c_flags_slice = glfw_c_flags.toOwnedSlice(b.allocator) catch @panic("oom");

    // Build ObjC flags (no ARC — GLFW uses manual retain/release)
    var glfw_m_flags = std.ArrayList([]const u8).empty;
    glfw_m_flags.append(b.allocator, "-D_GLFW_COCOA") catch @panic("oom");
    glfw_m_flags.append(b.allocator, "-fno-objc-arc") catch @panic("oom");
    glfw_m_flags.append(b.allocator, "-Wno-deprecated-declarations") catch @panic("oom");
    glfw_m_flags.append(b.allocator, "-Wno-nullability-completeness") catch @panic("oom");
    glfw_m_flags.appendSlice(b.allocator, c_flags) catch @panic("oom");
    const glfw_m_flags_slice = glfw_m_flags.toOwnedSlice(b.allocator) catch @panic("oom");

    glfw.addCSourceFiles(.{ .root = glfw_dep.path("."), .files = common_c, .flags = glfw_c_flags_slice });
    glfw.addCSourceFiles(.{ .root = glfw_dep.path("."), .files = macos_c, .flags = glfw_c_flags_slice });
    glfw.addCSourceFiles(.{ .root = glfw_dep.path("."), .files = macos_m, .flags = glfw_m_flags_slice });

    glfw.linkFramework("Cocoa");
    glfw.linkFramework("IOKit");
    glfw.linkFramework("CoreFoundation");
    glfw.linkFramework("CoreVideo");
    glfw.linkFramework("OpenGL");

    if (sysroot) |path| {
        const fw_path = b.fmt("{s}/System/Library/Frameworks", .{path});
        glfw.root_module.addFrameworkPath(.{ .cwd_relative = fw_path });
        const usr_lib = b.fmt("{s}/usr/lib", .{path});
        glfw.addLibraryPath(.{ .cwd_relative = usr_lib });
        // Add system include path for cups/ppd.h and other SDK headers
        const usr_include = b.fmt("{s}/usr/include", .{path});
        glfw.addSystemIncludePath(.{ .cwd_relative = usr_include });
    }

    return glfw;
}

fn addImguiLibrary(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    glfw_lib: *std.Build.Step.Compile,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
) *std.Build.Step.Compile {
    const glfw_dep = b.dependency("glfw", .{});
    const imgui_mod = createModule(b, target, optimize, true, optimize == .ReleaseFast, framework_path, private_framework_path);
    const imgui = b.addLibrary(.{
        .name = "imgui",
        .root_module = imgui_mod,
        .linkage = .static,
    });

    imgui.addIncludePath(b.path("third_party/imgui"));
    imgui.addIncludePath(glfw_dep.path("include"));

    const imgui_sources = &[_][]const u8{
        "third_party/imgui/imgui.cpp",
        "third_party/imgui/imgui_demo.cpp",
        "third_party/imgui/imgui_draw.cpp",
        "third_party/imgui/imgui_tables.cpp",
        "third_party/imgui/imgui_widgets.cpp",
        "third_party/imgui/backends/imgui_impl_glfw.cpp",
        "third_party/imgui/backends/imgui_impl_opengl3.cpp",
    };

    imgui.addCSourceFiles(.{ .files = imgui_sources, .flags = cxx_flags });
    imgui.linkLibrary(glfw_lib);

    return imgui;
}

fn addBenchmarks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml: *std.Build.Step.Compile,
    llama_core: *std.Build.Step.Compile,
    lfg_core: *std.Build.Step.Compile,
    spdlog_include: std.Build.LazyPath,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    sysroot: ?[]const u8,
) void {
    const bench_names = [_][]const u8{
        "benchmark_json_schema",
        "benchmark_healing_structured",
        "benchmark_tool_healing",
    };

    const bench_files = [_][]const u8{
        "src/benchmarks/benchmark_json_schema.cpp",
        "src/benchmarks/benchmark_healing_structured.cpp",
        "src/benchmarks/benchmark_tool_healing.cpp",
    };

    var bench_step = b.step("bench", "Build benchmarks");

    for (bench_names, 0..) |name, i| {
        const exe = addExe(b, target, optimize, name, &[_][]const u8{bench_files[i]}, spdlog_include, cxx_flags, framework_path, private_framework_path);
        exe.linkLibrary(lfg_core);
        exe.linkLibrary(ggml);
        addCommonExeLinks(exe, target, framework_path, private_framework_path, sysroot);
        b.installArtifact(exe);
        bench_step.dependOn(&exe.step);
    }

    // benchmark_perf_compare: links both lfg_core and llama_core for head-to-head comparison
    const perf_compare = addExe(b, target, optimize, "benchmark_perf_compare", &[_][]const u8{"src/benchmarks/benchmark_perf_compare.cpp"}, spdlog_include, cxx_flags, framework_path, private_framework_path);
    perf_compare.addIncludePath(b.path("third_party/llama.cpp/include"));
    perf_compare.addIncludePath(b.path("src/ggml"));
    perf_compare.linkLibrary(lfg_core);
    perf_compare.linkLibrary(llama_core);
    perf_compare.linkLibrary(ggml);
    addCommonExeLinks(perf_compare, target, framework_path, private_framework_path, sysroot);
    b.installArtifact(perf_compare);
    bench_step.dependOn(&perf_compare.step);
}

fn addTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml: *std.Build.Step.Compile,
    lfg_core: *std.Build.Step.Compile,
    spdlog_include: std.Build.LazyPath,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    sysroot: ?[]const u8,
) void {
    const tests = [_][]const u8{
        "test_loader",
        "test_inference",
        "test_c_api_errors",
        "test_json_sampling",
        "test_tool_calling",
        "test_350m_exhaustive_tooling",
        "test_lfm25_thinking_exhaustive_tooling",
        "test_lfm25_thinking_chat_tooling",
        "test_checkpointing",
        "test_leap_bundle",
        "test_token_healing",
        "test_healing_integration",
        "test_structured_checkpointing",
        "test_structured_double_accept",
        "test_reasoning_healing_integration",
        "test_complex_reasoning_healing",
        "test_reasoning_budget",
        "test_reasoning_gate",
        "test_model_capabilities",
        "test_parity",
        "test_softmax_safety",
        "test_grammar_completion",
        "test_stop_sequences",
        "test_json_schema_to_grammar",
        "test_session_lifecycle",
        "test_checkpoint_state",
        "test_max_tokens_reasoning",
        "test_tool_ranker",
        "test_entropy_monitor",
        "test_confidence_monitor",
        "test_surprise_monitor",
        "test_generate_loop",
        "test_structured_generate",
        "test_chat_integration",
        "test_tool_injection",
        "test_tool_chat_integration",
        "test_chat_first_message",
        "test_tool_call_parser",
        "test_bert_embedding",
    };

    const test_files = [_][]const u8{
        "src/tests/test_loader.cpp",
        "src/tests/test_inference.cpp",
        "src/tests/test_c_api_errors.cpp",
        "src/tests/test_json_sampling.cpp",
        "src/tests/test_tool_calling.cpp",
        "src/tests/test_350m_exhaustive_tooling.cpp",
        "src/tests/test_lfm25_thinking_exhaustive_tooling.cpp",
        "src/tests/test_lfm25_thinking_chat_tooling.cpp",
        "src/tests/test_checkpointing.cpp",
        "src/tests/test_leap_bundle.cpp",
        "src/tests/test_token_healing.cpp",
        "src/tests/test_healing_integration.cpp",
        "src/tests/test_structured_checkpointing.cpp",
        "src/tests/test_structured_double_accept.cpp",
        "src/tests/test_reasoning_healing_integration.cpp",
        "src/tests/test_complex_reasoning_healing.cpp",
        "src/tests/test_reasoning_budget.cpp",
        "src/tests/test_reasoning_gate.cpp",
        "src/tests/test_model_capabilities.cpp",
        "src/tests/test_parity.cpp",
        "src/tests/test_softmax_safety.cpp",
        "src/tests/test_grammar_completion.cpp",
        "src/tests/test_stop_sequences.cpp",
        "src/tests/test_json_schema_to_grammar.cpp",
        "src/tests/test_session_lifecycle.cpp",
        "src/tests/test_checkpoint_state.cpp",
        "src/tests/test_max_tokens_reasoning.cpp",
        "src/tests/test_tool_ranker.cpp",
        "src/tests/test_entropy_monitor.cpp",
        "src/tests/test_confidence_monitor.cpp",
        "src/tests/test_surprise_monitor.cpp",
        "src/tests/test_generate_loop.cpp",
        "src/tests/test_structured_generate.cpp",
        "src/tests/test_chat_integration.cpp",
        "src/tests/test_tool_injection.cpp",
        "src/tests/test_tool_chat_integration.cpp",
        "src/tests/test_chat_first_message.cpp",
        "src/tests/test_tool_call_parser.cpp",
        "src/tests/test_bert_embedding.cpp",
    };

    var test_step = b.step("test", "Build and run tests");

    for (tests, 0..) |name, i| {
        const exe = addExe(b, target, optimize, name, &[_][]const u8{test_files[i]}, spdlog_include, cxx_flags, framework_path, private_framework_path);
        exe.addIncludePath(b.path("src/tests"));
        exe.addIncludePath(b.path("src"));
        exe.addIncludePath(b.path("src/inference"));
        exe.addIncludePath(b.path("src/loader"));
        exe.addIncludePath(spdlog_include);
        exe.linkLibrary(lfg_core);
        if (std.mem.eql(u8, name, "test_reasoning_healing_integration") or std.mem.eql(u8, name, "test_complex_reasoning_healing") or std.mem.eql(u8, name, "test_reasoning_budget") or std.mem.eql(u8, name, "test_reasoning_gate")) {
            exe.linkLibrary(ggml);
        }
        addCommonExeLinks(exe, target, framework_path, private_framework_path, sysroot);
        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        if (std.mem.eql(u8, name, "test_leap_bundle")) {
            run.addArg("models/lfm2-350M.gguf");
        }
        if (std.mem.eql(u8, name, "test_model_capabilities")) {
            run.addArg("--model");
            run.addArg("models/lfm2-350M.gguf");
        }
        if (std.mem.eql(u8, name, "test_json_sampling")) {
            run.addArg("models/lfm2-350M.gguf");
        }
        test_step.dependOn(&run.step);
    }
}

fn addExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    name: []const u8,
    sources: []const []const u8,
    spdlog_include: std.Build.LazyPath,
    cxx_flags: []const []const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
) *std.Build.Step.Compile {
    const exe_mod = createModule(b, target, optimize, null, optimize == .ReleaseFast, framework_path, private_framework_path);
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = exe_mod,
    });
    if (framework_path) |path| {
        exe.root_module.addFrameworkPath(.{ .cwd_relative = path });
    }
    if (private_framework_path) |path| {
        exe.root_module.addFrameworkPath(.{ .cwd_relative = path });
    }
    exe.addCSourceFiles(.{ .files = sources, .flags = cxx_flags });
    exe.addIncludePath(b.path("src/ggml"));
    exe.addIncludePath(b.path("third_party/llama.cpp/include"));
    exe.addIncludePath(b.path("third_party/llama.cpp/src"));
    exe.addIncludePath(b.path("third_party/llama.cpp/common"));
    exe.addIncludePath(b.path("third_party/llama.cpp/vendor"));
    exe.addIncludePath(b.path("src/inference"));
    exe.addIncludePath(b.path("src/loader"));
    exe.addIncludePath(b.path("src/vision"));
    exe.addIncludePath(spdlog_include);
    return exe;
}

fn addCommonExeLinks(
    exe: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    sysroot: ?[]const u8,
) void {
    if (target.result.os.tag == .linux) {
        exe.linkSystemLibrary("dl");
    }
    if (target.result.os.tag != .windows) {
        exe.linkSystemLibrary("pthread");
    }
    if (target.result.os.tag == .macos) {
        if (framework_path) |path| {
            exe.root_module.addFrameworkPath(.{ .cwd_relative = path });
        }
        if (private_framework_path) |path| {
            exe.root_module.addFrameworkPath(.{ .cwd_relative = path });
        }
        if (sysroot) |path| {
            const usr_lib = exe.step.owner.fmt("{s}/usr/lib", .{path});
            exe.addLibraryPath(.{ .cwd_relative = usr_lib });
        }
        exe.linkFramework("Foundation");
        exe.linkFramework("Metal");
        exe.linkFramework("MetalKit");
        exe.linkFramework("Accelerate");
    }
}

fn collectFiles(allocator: std.mem.Allocator, dir: []const u8, exts: []const []const u8) ![][]const u8 {
    var list = std.ArrayList([]const u8).empty;
    var cwd = std.fs.cwd();
    var dir_handle = try cwd.openDir(dir, .{ .iterate = true });
    defer dir_handle.close();
    var walker = try dir_handle.walk(allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        const ext = std.fs.path.extension(entry.path);
        var matched = false;
        for (exts) |want| {
            if (std.mem.eql(u8, ext, want)) {
                matched = true;
                break;
            }
        }
        if (!matched) continue;

        const joined = try std.fs.path.join(allocator, &[_][]const u8{ dir, entry.path });
        try list.append(allocator, joined);
    }

    return list.toOwnedSlice(allocator);
}

fn targetLabel(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    return b.fmt("{s}-{s}", .{ @tagName(target.result.os.tag), @tagName(target.result.cpu.arch) });
}

fn resolveIsaTarget(
    b: *std.Build,
    base: std.Build.ResolvedTarget,
    cpu_model: *const std.Target.Cpu.Model,
    features_add: std.Target.Cpu.Feature.Set,
) std.Build.ResolvedTarget {
    var query = std.Target.Query.fromTarget(&base.result);
    query.cpu_model = .{ .explicit = cpu_model };
    query.cpu_features_add = features_add;
    return b.resolveTargetQuery(query);
}

fn getMacosSysroot(b: *std.Build) ?[]const u8 {
    const env_val = std.process.getEnvVarOwned(b.allocator, "SDKROOT") catch null;
    if (env_val) |path| {
        if (path.len > 0) return path;
    }

    var argv = [_][]const u8{ "xcrun", "--sdk", "macosx", "--show-sdk-path" };
    const result = std.process.Child.run(.{
        .allocator = b.allocator,
        .argv = &argv,
        .max_output_bytes = 16 * 1024,
    }) catch return null;
    if (result.term != .Exited or result.term.Exited != 0) return null;
    const trimmed = std.mem.trim(u8, result.stdout, " \r\n\t");
    if (trimmed.len == 0) return null;
    return b.allocator.dupe(u8, trimmed) catch @panic("oom");
}

fn extraFlagsForTarget(b: *std.Build, target: std.Build.ResolvedTarget) []const []const u8 {
    _ = b;
    _ = target;
    return &[_][]const u8{};
}

fn hasMingwLibs(b: *std.Build) bool {
    const zig_dir = std.fs.path.dirname(b.graph.zig_exe) orelse return false;
    const zig_root = std.fs.path.dirname(zig_dir) orelse return false;

    const x86_path = std.fs.path.join(b.allocator, &[_][]const u8{ zig_root, "x86_64-w64-mingw32", "lib" }) catch return false;
    const arm_path = std.fs.path.join(b.allocator, &[_][]const u8{ zig_root, "aarch64-w64-mingw32", "lib" }) catch return false;
    defer b.allocator.free(x86_path);
    defer b.allocator.free(arm_path);

    return pathExists(x86_path) and pathExists(arm_path);
}

fn pathExists(path: []const u8) bool {
    if (std.fs.cwd().access(path, .{})) |_| {
        return true;
    } else |_| {
        return false;
    }
}

fn makeFlags(
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    use_native: bool,
    enable_openmp: bool,
    extra: []const []const u8,
    sysroot: ?[]const u8,
) struct { c: []const []const u8, cxx: []const []const u8 } {
    var c_flags = std.ArrayList([]const u8).empty;
    var cxx_flags = std.ArrayList([]const u8).empty;

    c_flags.appendSlice(b.allocator, &[_][]const u8{"-std=c11"}) catch @panic("oom");
    cxx_flags.appendSlice(b.allocator, &[_][]const u8{"-std=c++17"}) catch @panic("oom");

    if (optimize != .Debug) {
        c_flags.appendSlice(b.allocator, &[_][]const u8{ "-O3", "-DNDEBUG" }) catch @panic("oom");
        cxx_flags.appendSlice(b.allocator, &[_][]const u8{ "-O3", "-DNDEBUG" }) catch @panic("oom");
        if (use_native) {
            c_flags.append(b.allocator, "-march=native") catch @panic("oom");
            cxx_flags.append(b.allocator, "-march=native") catch @panic("oom");
        }
    }

    c_flags.append(b.allocator, "-Wno-reserved-user-defined-literal") catch @panic("oom");
    cxx_flags.append(b.allocator, "-Wno-reserved-user-defined-literal") catch @panic("oom");

    if (enable_openmp) {
        c_flags.append(b.allocator, "-fopenmp") catch @panic("oom");
        cxx_flags.append(b.allocator, "-fopenmp") catch @panic("oom");
    }

    if (sysroot) |path| {
        c_flags.appendSlice(b.allocator, &[_][]const u8{ "-isysroot", path }) catch @panic("oom");
        cxx_flags.appendSlice(b.allocator, &[_][]const u8{ "-isysroot", path }) catch @panic("oom");
        const framework_path = b.fmt("{s}/System/Library/Frameworks", .{path});
        c_flags.appendSlice(b.allocator, &[_][]const u8{ "-F", framework_path }) catch @panic("oom");
        cxx_flags.appendSlice(b.allocator, &[_][]const u8{ "-F", framework_path }) catch @panic("oom");
        const private_framework_path = b.fmt("{s}/System/Library/PrivateFrameworks", .{path});
        c_flags.appendSlice(b.allocator, &[_][]const u8{ "-F", private_framework_path }) catch @panic("oom");
        cxx_flags.appendSlice(b.allocator, &[_][]const u8{ "-F", private_framework_path }) catch @panic("oom");
        const stub_path = b.pathFromRoot("third_party/macos_sdk_stubs");
        c_flags.appendSlice(b.allocator, &[_][]const u8{ "-I", stub_path }) catch @panic("oom");
        cxx_flags.appendSlice(b.allocator, &[_][]const u8{ "-I", stub_path }) catch @panic("oom");
        cxx_flags.append(b.allocator, "-Wno-elaborated-enum-base") catch @panic("oom");
    }

    if (extra.len > 0) {
        c_flags.appendSlice(b.allocator, extra) catch @panic("oom");
        cxx_flags.appendSlice(b.allocator, extra) catch @panic("oom");
    }

    return .{
        .c = c_flags.toOwnedSlice(b.allocator) catch @panic("oom"),
        .cxx = cxx_flags.toOwnedSlice(b.allocator) catch @panic("oom"),
    };
}

const IsaVariant = struct {
    name: []const u8,
    cpu_model: *const std.Target.Cpu.Model,
    features_add: std.Target.Cpu.Feature.Set = .empty,
};

fn addIsaCombinedLibraries(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    enable_metal: bool,
    embed_metal: bool,
    enable_accelerate: bool,
    enable_openmp: bool,
    sched_max_copies: u32,
    sched_max_backends: u32,
    spdlog_include: std.Build.LazyPath,
    spdlog_level: u32,
    shared_link_args: []const []const u8,
    target_label: []const u8,
    sysroot: ?[]const u8,
    framework_path: ?[]const u8,
    private_framework_path: ?[]const u8,
    windows_shared: bool,
    cpu_only: bool,
) void {
    const arch = target.result.cpu.arch;

    const x86_variants = [_]IsaVariant{
        .{ .name = "avx2", .cpu_model = &std.Target.x86.cpu.x86_64_v3 },
        .{ .name = "avx512", .cpu_model = &std.Target.x86.cpu.x86_64_v4 },
        .{ .name = "amx", .cpu_model = &std.Target.x86.cpu.sapphirerapids },
    };

    const arm_dotprod_features = std.Target.aarch64.featureSet(&[_]std.Target.aarch64.Feature{.dotprod});
    const arm_i8mm_features = std.Target.aarch64.featureSet(&[_]std.Target.aarch64.Feature{ .i8mm, .dotprod });

    const arm_variants_macos = [_]IsaVariant{
        .{ .name = "dotprod", .cpu_model = &std.Target.aarch64.cpu.apple_m1, .features_add = arm_dotprod_features },
        .{ .name = "i8mm", .cpu_model = &std.Target.aarch64.cpu.apple_m1, .features_add = arm_i8mm_features },
    };

    const arm_variants = [_]IsaVariant{
        .{ .name = "dotprod", .cpu_model = &std.Target.aarch64.cpu.neoverse_n1, .features_add = arm_dotprod_features },
        .{ .name = "i8mm", .cpu_model = &std.Target.aarch64.cpu.neoverse_v1, .features_add = arm_i8mm_features },
    };

    const arm_variants_windows = [_]IsaVariant{
        .{ .name = "dotprod", .cpu_model = &std.Target.aarch64.cpu.neoverse_n1, .features_add = arm_dotprod_features },
    };

    const variants = switch (arch) {
        .x86_64, .x86 => x86_variants[0..],
        .aarch64, .arm => if (target.result.os.tag == .macos) arm_variants_macos[0..] else if (target.result.os.tag == .windows) arm_variants_windows[0..] else arm_variants[0..],
        else => return,
    };

    for (variants) |variant| {
        const variant_target = resolveIsaTarget(b, target, variant.cpu_model, variant.features_add);
        const flags = makeFlags(b, optimize, false, enable_openmp, extraFlagsForTarget(b, variant_target), sysroot);
        const ggml = addGgmlLibrary(
            b,
            variant_target,
            optimize,
            enable_metal,
            embed_metal,
            enable_accelerate,
            enable_openmp,
            sched_max_copies,
            sched_max_backends,
            flags.c,
            flags.cxx,
            framework_path,
            private_framework_path,
            cpu_only,
        );
        const lfg_vision = addLiquidVision(b, variant_target, optimize, ggml, spdlog_include, spdlog_level, flags.cxx, framework_path, private_framework_path);
        const lfg_core = addLiquidCore(b, variant_target, optimize, ggml, lfg_vision, spdlog_include, spdlog_level, flags.cxx, framework_path, private_framework_path);

        const suffix = b.fmt("-{s}-{s}", .{ target_label, variant.name });
        addCombinedLibrary(b, ggml, lfg_core, lfg_vision, variant_target, optimize, suffix, shared_link_args, windows_shared);
    }
}

fn makeSharedLinkArgs(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    enable_metal: bool,
    enable_accelerate: bool,
    enable_openmp: bool,
    sysroot: ?[]const u8,
) []const []const u8 {
    var args = std.ArrayList([]const u8).empty;

    if (sysroot) |path| {
        args.appendSlice(b.allocator, &[_][]const u8{ "-isysroot", path }) catch @panic("oom");
        const framework_path = b.fmt("{s}/System/Library/Frameworks", .{path});
        args.appendSlice(b.allocator, &[_][]const u8{ "-F", framework_path }) catch @panic("oom");
        const private_framework_path = b.fmt("{s}/System/Library/PrivateFrameworks", .{path});
        args.appendSlice(b.allocator, &[_][]const u8{ "-F", private_framework_path }) catch @panic("oom");
        const usr_lib_path = b.fmt("{s}/usr/lib", .{path});
        args.appendSlice(b.allocator, &[_][]const u8{ "-L", usr_lib_path }) catch @panic("oom");
        const usr_lib_system_path = b.fmt("{s}/usr/lib/system", .{path});
        args.appendSlice(b.allocator, &[_][]const u8{ "-L", usr_lib_system_path }) catch @panic("oom");
    }

    if (target.result.os.tag == .macos) {
        if (enable_metal) {
            args.appendSlice(b.allocator, &[_][]const u8{ "-framework", "Foundation" }) catch @panic("oom");
            args.appendSlice(b.allocator, &[_][]const u8{ "-framework", "Metal" }) catch @panic("oom");
            args.appendSlice(b.allocator, &[_][]const u8{ "-framework", "MetalKit" }) catch @panic("oom");
        }
        if (enable_accelerate) {
            args.appendSlice(b.allocator, &[_][]const u8{ "-framework", "Accelerate" }) catch @panic("oom");
        }
    } else if (target.result.os.tag == .linux) {
        args.append(b.allocator, "-ldl") catch @panic("oom");
    } else if (target.result.os.tag == .windows) {
        args.append(b.allocator, "-Wl,--export-all-symbols") catch @panic("oom");
    }

    if (target.result.os.tag != .windows) {
        args.append(b.allocator, "-lpthread") catch @panic("oom");
        args.appendSlice(b.allocator, &[_][]const u8{ "-lc++", "-lc++abi", "-lc" }) catch @panic("oom");
    }
    if (enable_openmp) {
        args.append(b.allocator, "-lomp") catch @panic("oom");
    }

    return args.toOwnedSlice(b.allocator) catch @panic("oom");
}
