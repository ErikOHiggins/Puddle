# Test requests and hardware wanted

Puddle is built and measured on one machine, echo (two RTX 5060 Ti 16 GB, Linux). Results from other hardware are
welcome: they tell us what setup really does and show where the engine picks something wrong. Write a report as
described in [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md): the build, the card and PCIe width, the CPU and RAM
speed, the model file, the exact command, three runs per size, and what you did not test. Post it in an issue.

An item stays until a report answers it.

| Wanted | Why |
|---|---|
| Any of Puddle's models on Windows | Laguna and Occamy have only run on Linux |
| Any of Puddle's models on an AMD card | strata-poly compiles for AMD, but none of its models has run on one |
| Laguna-S UD-Q4_K_XL or poolside's Q4_K_M | Only UD-Q3_K_XL has run; Q4_K_M keeps 8 expert layers in BF16 on the processor |
| Occamy APEX-I-Compact or APEX-Balanced | Only APEX-Quality has run |
| One card of 16 GB or more, or one of 24 GB | Every Laguna run so far was split over two 16 GB cards |
| A Mac with Apple Silicon | The Metal build has not been run in Puddle; Laguna is not in it yet |
| A long soak (an hour or more of back-to-back requests) | Shows crashes, drift and heat that a short run does not |

To add a request, open a pull request that adds a row, or say so in an issue.
