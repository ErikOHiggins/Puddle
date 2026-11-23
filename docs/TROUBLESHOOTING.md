# Something went wrong?

The common problems and what to do. Back to the [README](../README.md#something-went-wrong).

## While installing or starting

**It stopped while downloading, compiling or installing.**
Run `START-HERE.bat` (Linux: `./setup.sh`) again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run setup again.
A driver older than 580 also works with CUDA 12: `./setup.sh --setup --cuda 12` ([OLDER_GPUS.md](OLDER_GPUS.md)).

**It says port 8080 is already in use.**
Puddle is already running: look for its window. Or another program uses the port: `./setup.sh --port 8081`. Only an
address that is really taken says "already in use"; any other reason (Windows keeps the port reserved, or the config's
`"host"` is not an address of this PC) is printed as what the OS said, with what to change.

**"Windows blocked the engine (or the image encoder) ... Smart App Control".**
On a clean Windows 11, Smart App Control can refuse programs that are not signed, and `strata-poly.exe` and
`strata-vision.exe` are not. Turn it off (Windows Security > App & browser control > Smart App Control settings; it
cannot be turned back on without a reset of Windows), or, if only the image encoder is blocked, run
`START-HERE.bat --setup --vision no`.

**Python or the build tools could not be installed.**
Install what it names (links are printed), then run it again. Everything already done is kept.

**Linux: the engine does not compile (`unsupported GNU version`, or `exception specification is incompatible` for
`cospi`/`sinpi`/`rsqrt`).** Two known mismatches between the CUDA toolkit and a new Linux:
- gcc newer than 14 (Ubuntu 26.04's default 15): CUDA 12.x and 13.0 refuse it. Install g++-14 and run
  `CXX=g++-14 ./setup.sh` (CC, CXX and CUDAHOSTCXX name the build's compilers; setup passes them to CMake).
- glibc 2.43 with CUDA 12.9: the toolkit's math headers clash with glibc's. Use CUDA 12.8 or 13.x instead. Setup takes
  the newest toolkit it finds; `STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc ./setup.sh` makes it use that one.

**An RTX 50 card answers with nonsense, or reads prompts wrongly.**
CUDA 13.2.0 and 13.2.1's compiler (nvcc build 13.2.51) miscompiles some quantized kernels for sm_120. Use CUDA 13.0,
13.1 or 13.2.2 (build 13.2.86 fixes it; `STRATA_NVCC=<path to its nvcc>` picks one). Setup warns when it finds 13.2.0
or 13.2.1 for such a card, and takes an older 13.x when one is installed.

**The first start takes minutes.**
The engine reads the model's file (20 to 96 GB) into RAM; the next starts are faster while the files are in the OS
file cache.

## While it answers

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM, and the experts are read from the disk. Close other programs, or pick a smaller size or
model.

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine:
`sudo dmesg | grep -i -E 'killed process|out of memory'`). Send your message again: Puddle starts the engine by itself.
If it keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, run setup again with a larger `--context`, or
let the server drop old turns ([API.md](API.md#context)).

**The answers are the same every time.**
The config has no `sampling` block (one made by hand), so a request that sends no temperature is answered greedy. Run
setup again for the model (it writes the model's sampling and keeps everything else you added), add a block by hand,
or set the sampling in your app ([API.md](API.md#sampling-and-limits)).

**It is slower than the README's numbers.**
The monitor plugged into the graphics card and other GPU programs take VRAM from the expert cache; RAM running below
its rated speed (enable EXPO/XMP in the BIOS) slows the share of the experts the CPU computes.

**An earlier reply that came back empty is gone from the chat history.**
The server leaves out finished assistant turns that have no text, so the next reply does not copy the empty ones.
`STRATA_KEEP_EMPTY_TURNS=1` in the server's environment renders them as before.

**The Windows display driver resets, then the PC blue-screens (0x141, then 0x116 in `nvlddmkm`) during an answer.**
That is the NVIDIA driver: PolyStrata's report showed the same with other CUDA programs on that driver. Try another
driver (a Studio one, or an older one) and a lower power limit.

**Pictures are refused, or slow.**
"this server was started without the vision encoder": the model was set up for text only; run setup again with
`--vision yes` (Occamy, Gemma 4 and Nemotron 3.5 Bolt). A picture that takes several seconds is read by Gemma 4's
encoder on the CPU, which setup chooses on a card under 15.5 GB; `--vision gpu` puts it on the card (1.6 GB of VRAM
for Occamy, 2.4 GB for Gemma 4).

## Nemotron 3.5 Bolt on llama-server (`nemotron-bolt-llama`)

**Setup says it runs with 30 GB of VRAM or more.**
The cards you chose have less together. With two cards, choose both (`--gpus 0,1`); with less VRAM, set Bolt up on
strata-poly (`--family nemotron-bolt`), which keeps the experts that do not fit in RAM.

**llama-server does not compile.**
Setup prints the build's output, and the build folder is `build-llama-server/`. Run setup again: it compiles only what
is missing. The compiler notes above apply to it too (gcc newer than 14: `CXX=g++-14 ./setup.sh`).

**"llama-server exited before it was ready".**
The message ends with llama-server's last log lines; the whole log is `strata-nemotron-llama-bolt-24gb.log` in the Puddle
folder. If it ran out of VRAM while loading, close the program that holds VRAM, or run
`./setup.sh --setup --family nemotron-bolt-llama --context N` with a smaller N.

**"llama-server read N prompt tokens where Puddle planned M".**
Puddle's count of a picture's tokens differs from llama-server's. The request is still answered (llama-server's count
is the one it goes by). Please report it with the picture's width and height.

## AMD cards

**"No AMD GPU found (the amdgpu driver's KFD topology is empty)" on Linux.**
The kernel's amdgpu driver is not loaded for the card. Setup lists every card it found and whether Puddle can use it.

**The engine stops at start with the card's name, its architecture and the build's list.**
The engine was compiled for another card (for example after moving the Puddle folder to another PC). Run
`./setup.sh --setup --backend hip`: it compiles the engine for this card's architecture.

**On Windows the engine exits with `0xC0000005` in `amdhip64_7.dll` before it prints anything.**
Two things in your environment can cause it, and the server repairs both before it starts the engine (it says so in
the log): `HIP_PATH`, `HIP_DEVICE_LIB_PATH` or `LLVM_PATH` naming a ROCm folder that no longer exists, and a
`TEMP`/`TMP` folder the engine cannot create files in. If you start the engine by hand, fix them yourself:
`set HIP_PATH=`, and point `TEMP` and `TMP` at a normal folder such as `C:\Temp`.

**Windows AMD: the driver resets (VIDEO_ENGINE_TIMEOUT_DETECTED / screen flicker / the engine dies mid-answer).**
Windows gives a GPU about 2 seconds to answer; when a graphics card does not, it resets the driver (a TDR): the screen
flickers, the engine dies, and the Event Viewer shows `VIDEO_ENGINE_TIMEOUT_DETECTED`. PolyStrata's users saw it on
gfx1201 cards while a long prompt was read. Setup points RX 9000 owners on Windows here. Try these in order:
1. **Send the evidence** in an issue: the last ~80 lines of `strata-<model>.log`, your AMD driver version (in AMD
   Software, under System), your card and Windows version, and the exact steps (context size, prompt length, which app
   sent it). If Windows wrote a dump, the newest `.dmp` in `C:\Windows\LiveKernelReports\WATCHDOG` helps; it holds no
   chats.
2. **Driver:** use the newest AMD Software release for your card (Adrenalin, not a Windows Update driver) and restart.
   If it started after a driver update, going back one version is a fair test.
3. **Optional, your decision: give Windows more time (`TdrDelay`).** In the registry,
   `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`, a DWORD `TdrDelay` of `10` (seconds) lets
   a long GPU job finish before Windows resets the driver; restart the PC. It hides the symptom and does not fix a GPU
   that has really hung. Puddle changes no Windows setting itself; to undo it, delete the value.

## Still stuck?

Open an [issue](https://github.com/ErikOHiggins/Puddle/issues) and attach `strata-<model>.log` from the Puddle folder.
