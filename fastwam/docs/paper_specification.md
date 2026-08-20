# Ledger\-WAM: Causal Debt\-Aware World Action Models for Self\-Healing Long\-Horizon Planning

# 题目

# 摘要

World Action Models（WAMs）通过联合建模动作条件下的世界动态与动作生成，正在成为具身智能中的一种重要范式。然而，现有 WAM 在长时程操作任务中仍然十分脆弱：随着想象 rollout 变长，接触状态、抓取状态、遮挡关系和物体持续性上的微小预测误差会在后续规划中不断累积，最终导致任务失败。本文认为，这一问题并不只是未来预测不准确，而是 WAM 在动作条件想象过程中不断积累了未经验证的因果假设。为此，我们提出 **Ledger\-WAM**，一种用于自修复长时程规划的可回滚因果信念 World Action Model。不同于将想象未来表示为密集视频帧或单一潜变量状态，Ledger\-WAM 维护一个 **causal belief ledger**，其中显式记录关于物体、关系、前置条件、动作效果、观测证据、不确定性以及下游依赖的动作条件因果声明。该 ledger 使模型能够识别未来动作计划依赖了哪些想象事实，并为每个因果声明分配一个 **causal debt**，用于衡量在未经验证的情况下继续规划的风险。在此基础上，我们提出一种 **self\-healing planner**，能够主动选择最小代价的修复动作，以验证、推翻或修正高债务因果声明，然后再继续执行任务。不同于标准重规划方法在预测与真实观测不一致后重新生成完整动作序列，我们的方法能够定位潜在失败所对应的具体因果假设，并通过局部回滚实现有针对性的信念修复。在包含遮挡、接触丰富交互、物体持续性和分支子目标的长时程操作任务中，Ledger\-WAM 相比密集未来预测式 WAM、层级 world model 以及基于 verifier 的自适应重规划方法，在任务成功率、失败定位和修复效率上均取得更优表现。我们的结果表明，可靠的 WAM 长时程具身规划不仅需要想象未来观测，更需要追踪并修复这些想象未来所依赖的因果假设。

# 关键字

# Introduction

1\.2 中涉及的代表性工作，我按这些论文的完整题名来组织：**World Action Models: The Next Frontier in Embodied AI**、**H\-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model**、**When to Trust Imagination: Adaptive Action Execution for World Action Models**、**Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation**、**Fast\-WAM: Do World Action Models Need Test\-time Future Imagination?**、**GigaWorld\-Policy: An Efficient Action\-Centered World\-\-Action Model**、**Being\-H0\.7: A Latent World\-Action Model from Egocentric Videos**。这些论文分别对应 WAM 定义、层级规划、自适应执行、结构化未来预测、推理效率和潜空间 WAM 等方向。\([arXiv](https://arxiv.org/abs/2605.12090?utm_source=chatgpt.com)\)

## 1\.1 课题的研究意义

World Action Model（WAM）是具身智能和机器人学习中正在兴起的重要研究方向。与传统 Vision\-Language\-Action 模型直接从观测和语言指令映射到动作不同，WAM 强调在动作生成过程中显式建模动作条件下的世界变化，使模型不仅知道“应该做什么”，还能够预测“做了之后世界会如何变化”。这种能力对于长时程操作任务尤为关键，因为真实机器人任务往往不是单步反应式控制，而是由多个相互依赖的子目标构成，例如抓取物体、打开容器、移动目标、放置物体以及在失败后重新调整策略。

然而，当前 WAM 在长时程规划中仍然面临严重的不稳定性。随着想象 rollout 的时间跨度增加，模型在接触状态、抓取状态、遮挡关系、物体持续性和空间关系上的微小误差会不断传递到后续动作决策中，最终导致计划漂移和任务失败。尤其在真实机器人操作场景中，许多关键状态并不能被视觉直接观测到，例如物体是否真正被夹爪稳定抓住、抽屉是否已经完全打开、被遮挡物体是否仍在原位置等。模型如果在这些状态上形成错误假设，后续规划就会建立在不可靠的基础之上。

因此，本课题的研究意义在于：从根本上提升 WAM 在长时程具身规划中的可靠性、可解释性和自修复能力。我们关注的不仅是如何生成更逼真的未来观测，也不仅是如何提高动作预测精度，而是如何让 WAM 显式追踪其未来规划所依赖的关键因果假设，并在这些假设不可靠时主动验证和修复。该问题对于推动 WAM 从短程视觉预测模型走向可靠的长时程具身智能系统具有重要意义，也为机器人任务规划、模型化强化学习、因果表征学习和主动感知提供了新的研究视角。

## 1\.2 之前的方法和存在的缺陷

近期的 **World Action Models: The Next Frontier in Embodied AI** 对 WAM 领域进行了系统定义，将其概括为联合建模预测性世界状态和动作生成的具身基础模型。该工作指出，WAM 的核心优势在于能够把世界动态预测纳入动作生成过程，从而弥补传统 VLA 模型缺乏动作后果建模的不足。然而，该综述也表明，当前 WAM 研究仍然处于早期阶段，不同方法在架构、训练目标、未来表示和评测协议上较为分散，尤其缺乏针对长时程规划可靠性的统一建模框架。

在层级规划方面，**H\-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model** 提出将高层逻辑世界模型与低层视觉世界模型结合起来，用于机器人任务与运动规划。该方法试图利用高层逻辑状态缓解长时程视觉 rollout 中的误差累积问题，并增强长任务执行的稳定性。然而，这类方法主要依赖预定义或可抽象的逻辑状态，其状态表达通常较为粗粒度，难以刻画真实操作中细粒度的接触、遮挡、滑落、夹持稳定性和物体持续性等关键因果条件。当任务失败时，模型也较难定位究竟是哪一个具体假设导致了后续计划失效。

在自适应执行方面，**When to Trust Imagination: Adaptive Action Execution for World Action Models** 将 WAM 的执行过程建模为未来想象与真实观测之间的一致性验证问题。该方法通过 verifier 判断当前想象 rollout 是否仍然可信，并据此动态调整动作块长度，在可靠阶段执行更长动作序列，在不可靠阶段提前重新规划。该方向有效缓解了固定动作块执行带来的盲目性，但其主要判断依据仍然是整体未来预测与真实观测之间是否一致。当出现偏差时，模型通常只能决定是否重新规划，却难以明确指出偏差背后的具体因果假设，例如“物体是否已经抓稳”“容器是否已经打开”或“被遮挡物体是否仍然存在”。因此，该类方法具备执行层面的自适应性，但缺乏假设级别的错误归因和局部修复能力。

在结构化未来预测方面，**Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation** 提出 StructVLA，用稀疏且具有物理意义的结构化帧替代密集未来视频预测。该方法认为密集视觉未来存在大量冗余，并且会在长时程中造成误差累积和计划漂移，因此改为预测与任务进展密切相关的运动学里程碑。该思路显著提升了未来表示与低层控制之间的对齐程度，但其核心仍然是对未来轨迹或关键状态的预测，并未显式记录计划对哪些因果条件产生依赖。当某个中间状态没有被真实验证时，模型仍然缺乏一种机制来评估继续规划的风险，也缺乏针对该状态的主动验证和回滚机制。

在推理效率方面，**Fast\-WAM: Do World Action Models Need Test\-time Future Imagination?** 重新审视了 WAM 是否必须在测试时显式生成未来。该工作表明，视频建模在训练阶段对表征学习具有重要作用，但测试时的显式未来生成并不总是必要，因此可以跳过未来帧生成以提高部署效率。类似地，**GigaWorld\-Policy: An Efficient Action\-Centered World\-\-Action Model** 提出以动作为中心的 WAM，通过解耦未来视频生成和动作解码来降低推理开销，并减少动作预测对未来视频质量的依赖。这些方法有效提升了 WAM 的实时性和工程可部署性，但它们主要解决的是推理效率问题，而不是长时程规划中的因果假设累积问题。模型即使运行更快，仍可能在遮挡、接触和物体状态变化等关键环节形成错误信念，并将这些错误继续传递到后续动作中。

在潜空间建模方面，**Being\-H0\.7: A Latent World\-Action Model from Egocentric Videos** 提出通过潜在推理空间引入未来感知能力，避免在推理阶段显式生成未来视频。该方法说明，WAM 的价值可以通过紧凑的潜空间表征传递给动作生成模块，而不必依赖像素级未来 rollout。然而，潜空间表征通常缺乏可解释的因果结构，难以直接说明模型当前相信哪些对象关系、哪些动作效果已经发生、哪些状态只是未经验证的假设。因此，当长时程任务失败时，潜空间 WAM 虽然可能具备较强的预测能力，却仍然难以进行假设级别的错误定位、风险评估和主动修复。

总体来看，已有 WAM 方法主要从三个角度缓解长时程规划问题：一是通过层级抽象减少长距离预测难度，二是通过结构化未来或潜空间表示降低密集视觉 rollout 的冗余和漂移，三是通过自适应执行或高效动作解码提升部署效率。然而，这些方法普遍缺少一种机制来显式表示和管理“未来规划所依赖但尚未被验证的因果假设”。也就是说，当前 WAM 往往能够想象未来、生成动作或判断整体 rollout 是否可信，但难以回答更关键的问题：当前计划到底依赖了哪些想象事实，哪些事实尚未被真实观测确认，哪些假设一旦错误会导致任务失败，以及机器人应该采取什么动作来验证或修复这些假设。

## 1\.3 我们的方法和获得的改进

为解决上述问题，我们提出 **Ledger\-WAM: Causal Debt\-Aware World Action Models for Self\-Healing Long\-Horizon Planning**。该方法的核心思想是将 WAM 的长时程想象过程从“未来帧序列预测”转变为“因果信念账本维护”。具体而言，Ledger\-WAM 不再仅仅预测未来观测或潜在状态，而是维护一个 causal belief ledger，用于显式记录动作条件下形成的关键因果声明。这些声明包括物体之间的关系、动作执行前提、动作产生的效果、当前观测证据、不确定性、下游依赖以及回滚位置。通过这种方式，模型能够清楚地知道当前计划建立在哪些因果假设之上，并评估哪些假设在未经验证的情况下继续用于规划会带来较高风险。

我们进一步提出 causal debt 的概念，用于衡量某个因果声明在当前任务中的风险程度。一个因果声明的风险不仅取决于模型对它的置信度，还取决于它对后续动作的影响、当前是否可观测、错误后是否容易修复以及是否会引发连锁失败。例如，在抓取后移动物体的任务中，“物体已经稳定处于夹爪中”就是一个高风险因果声明。如果该声明没有被可靠验证，后续移动、放置或倾倒动作都可能建立在错误基础之上。通过 causal debt，Ledger\-WAM 能够把长时程规划中的不确定性从整体轨迹层面细化到具体因果假设层面，从而实现更精确的风险识别和错误归因。

在规划机制上，我们提出 self\-healing planner，使机器人在继续执行任务动作之前，能够主动选择最小代价的修复动作来验证、推翻或修正高风险因果声明。与已有自适应重规划方法不同，self\-healing planner 不只是发现想象与现实不一致后重新生成完整动作序列，而是首先定位导致潜在失败的具体因果假设，然后通过局部验证、局部修复和局部回滚来恢复可靠的信念状态。例如，当模型无法确定物体是否稳定抓住时，它可以主动执行轻微抬升、调整视角、触觉确认或重新抓取等修复动作，而不是盲目继续执行后续长程计划。这种机制使 WAM 具备了类似“自我检查”和“自我修复”的能力。

相较于已有方法，Ledger\-WAM 预期在三个方面带来改进。第一，在长时程任务成功率方面，因果信念账本能够减少未验证假设在后续规划中的传播，从而降低由于抓取失败、遮挡误判、物体滑落或状态漂移导致的任务失败。第二，在失败定位方面，模型能够指出具体是哪一个因果声明导致计划不可靠，而不是仅仅给出整体 rollout 是否可信的判断。第三，在修复效率方面，self\-healing planner 可以通过小规模局部修复替代完整重规划，使机器人在复杂操作任务中以更低的代价恢复任务执行。该方法尤其适用于包含遮挡、接触丰富交互、物体持续性和分支子目标的长时程操作场景。

## 1\.4 贡献总结

本文围绕 WAM 在长时程具身规划中的误差累积和可靠性不足问题，提出了一种新的问题视角和方法框架。我们的核心观点是，WAM 的长时程失败并不只是未来预测不准确，而是模型在动作条件想象过程中积累了大量未经验证的因果假设，并在后续规划中持续依赖这些假设。基于这一观点，本文提出 Ledger\-WAM，将未来想象表示为可追踪、可验证、可回滚的因果信念账本，从而为 WAM 提供了一种更适合长时程规划的内部世界表示。

本文的第一项贡献是提出 causal belief ledger，用于显式建模动作条件下的对象关系、前置条件、动作效果、观测证据、不确定性和下游依赖。该表示使 WAM 能够从整体未来预测转向假设级别的因果信念维护，为长时程规划中的错误定位和可靠性分析提供了新的基础。

本文的第二项贡献是提出 causal debt，用于刻画未经验证的因果声明对后续计划造成的潜在风险。通过 causal debt，模型能够识别哪些想象事实最需要被验证，哪些假设一旦错误最可能导致任务失败，从而将不确定性建模从轨迹级别细化到因果声明级别。

本文的第三项贡献是提出 self\-healing planner，使 WAM 能够在执行过程中主动选择修复动作，对高风险因果声明进行验证、修正或回滚。该机制区别于传统重规划方法，不是简单地在失败后重新生成动作序列，而是对导致潜在失败的局部因果假设进行针对性修复，从而提升长时程任务执行的稳定性和效率。

本文的第四项贡献是在长时程操作任务中验证 Ledger\-WAM 的有效性。通过包含遮挡、接触丰富交互、物体持续性和分支子目标的实验设置，我们展示该方法相较于密集未来预测式 WAM、层级世界模型、结构化未来预测方法和基于 verifier 的自适应重规划方法，在任务成功率、失败定位能力和修复效率方面均具有潜在优势。整体而言，本文为 WAM 从短程动作预测走向可靠长时程具身规划提供了一种新的建模范式。

# 相关工作

## 2\.1 World Action Models 与动作条件世界建模

World Action Models（WAMs）近年来成为具身智能和机器人学习中的重要研究方向，其核心思想是在动作生成过程中同时建模动作条件下的世界动态。**World Action Models: The Next Frontier in Embodied AI** 系统定义了 WAM，将其概括为联合建模未来世界状态与动作生成的具身基础模型，并指出 WAM 是连接 Vision\-Language\-Action 模型与 world model 的新范式。**World Action Models are Zero\-shot Policies** 提出 DreamZero，将预训练视频扩散模型转化为 WAM，通过联合预测未来视频和动作实现跨任务、跨环境甚至跨 embodiment 的零样本泛化。**Cosmos Policy: Fine\-Tuning Video Models for Visuomotor Control and Planning** 将大规模视频模型直接微调为机器人策略，使动作、未来状态和价值估计都以潜在帧形式在同一视频模型中生成。**Causal World Modeling for Robot Control** 提出 LingBot\-VA，通过自回归扩散框架联合学习视频帧预测与策略执行，强调动作与视觉动态之间的因果联系。**Fast\-WAM: Do World Action Models Need Test\-time Future Imagination?** 研究 WAM 是否必须在测试阶段显式生成未来，并表明视频预测的主要价值可能来自训练阶段的世界表征学习而非测试时 rollout。**GigaWorld\-Policy: An Efficient Action\-Centered World\-\-Action Model** 提出以动作为中心的 WAM，将动作预测与未来视频生成解耦，使测试阶段可以跳过显式视频生成以提高推理效率。**Being\-H0\.7: A Latent World\-Action Model from Egocentric Videos** 使用未来感知的潜在查询将世界建模收益注入 VLA 策略，在推理阶段不生成未来帧也能获得预测式表征能力。**AIM: Intent\-Aware Unified world action Modeling with Spatial Value Maps** 引入空间价值图作为未来动态与动作解码之间的显式接口，使 WAM 能够更好地建模交互位置和操作意图。**GeoSem\-WAM: Geometry\- and Semantic\-Aware World Action Models** 在 RGB 未来预测之外加入几何和语义监督，使 WAM 的潜在表征具备更强的空间结构理解能力。**OA\-WAM: Object\-Addressable World Action Model for Robust Robot Manipulation** 将世界状态分解为机器人 slot 和对象 slot，使动作解码器能够稳定寻址语言指令中涉及的目标物体。**HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models** 通过预测专家和反应专家的自适应协调，尝试同时获得 WAM 的泛化能力和精细操作能力。总体来看，这些工作推动了 WAM 从密集未来视频预测逐渐发展到高效动作中心建模、潜空间建模、对象寻址建模和多模态世界表征，但它们大多仍然缺少对计划所依赖的关键因果假设的显式表达。

## 2\.2 长时程规划、自适应执行与误差累积

长时程规划是 WAM 走向真实具身智能系统必须解决的核心问题，因为随着任务长度增加，模型在视觉状态、接触关系、抓取稳定性和物体持续性上的微小误差会不断积累并影响后续动作。**H\-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model** 将高层逻辑世界模型与低层视觉世界模型结合，用层级状态转移缓解长时程机器人任务中的误差累积。**Hierarchical Planning with Latent World Models** 学习多个时间尺度的潜在世界模型，并在不同时间尺度之间进行层级规划，以降低长时程控制中的预测误差和搜索复杂度。**When to Trust Imagination: Adaptive Action Execution for World Action Models** 将 WAM 执行建模为未来想象与真实观测之间的一致性验证问题，并通过 verifier 动态决定继续执行还是提前重规划。**Dreaming when Necessary: Advancing World Action Models with Adaptive Multi\-Modal Reasoning** 提出 AdaWAM，根据任务阶段自适应触发文本推理或视觉推理，以提高复杂长时程任务中的效率和性能。**AdaWorldPolicy: World\-Model\-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation** 将世界模型、动作专家和力预测器结合，并通过在线自适应学习增强机器人在动态环境中的鲁棒性。**Compositional Planning with Jumpy World Models** 使用多步 dynamics 的 jumpy world model 对预训练策略进行时间抽象组合，从而缓解长时程预测和策略组合中的误差累积。**WIMLE: Uncertainty\-Aware World Models with IMLE for Sample\-Efficient Continuous Control** 通过多模态随机世界模型和不确定性感知加权减少 model\-based reinforcement learning 中的过度自信预测和 rollout 偏差。**World\-Model\-Augmented Web Agents with Action Correction** 将 world model 用作环境转移专家来辅助 action model 进行动作校正，说明预测式环境建模也可用于非机器人长时程 agent 执行。**WorldRFT: Latent World Model Planning with Reinforcement Fine\-Tuning** 在自动驾驶场景中结合潜在世界模型、层级规划和强化微调，展示了世界模型在长时程安全规划中的应用潜力。总体而言，这些方法从层级抽象、自适应执行、多模态推理和不确定性建模等角度缓解了长时程误差累积问题，但它们通常只能判断整体计划是否可靠，难以明确指出当前计划依赖了哪些未经验证的因果假设，也缺乏针对具体错误假设的局部修复机制。

## 2\.3 结构化表示、评测基准与可靠性分析

除了层级规划和自适应执行，近期研究也开始从结构化表示、控制导向表征和评测基准等角度重新审视 WAM 的可靠性问题。**Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation** 提出 StructVLA，用稀疏且具有运动学意义的结构化帧替代密集未来视频，以减少视觉冗余和长时程计划漂移。**WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models** 提出统一评测框架，同时衡量 embodied world model 的视觉质量和下游功能价值。**WorldArena 2\.0: Extending Embodied World Model Benchmarking on Modality, Functionality and Platform** 扩展了 WorldArena 的模态、功能和平台覆盖范围，使世界模型评测进一步从视觉保真度转向功能可用性。**World Model for Robot Learning: A Comprehensive Survey** 系统总结了机器人 world model 的表示、训练、规划和控制方法，并强调 action\-conditioned prediction 对机器人学习的重要性。**Wow, wo, val\! A Comprehensive Embodied World Model Evaluation Protocol** 提出更细粒度的 embodied world model 评测协议，强调仅用视频生成质量评估世界模型是不充分的。**Latent State Design for World Models under Sufficiency Constraints** 讨论潜在状态设计中的充分性约束，说明 world model 的 latent state 应保留对控制和预测真正必要的信息。**Object\-Centric World Models from Few\-Shot Annotations for Sample\-Efficient Reinforcement Learning** 探索少量标注下的对象中心世界模型，为 WAM 中对象级状态分解和因果关系建模提供了相关思路。**PhysWorld: Physical World Models for Robot Learning** 通过从生成视频中恢复物理世界模型并将运动 grounding 到机器人动作，强调视觉未来必须转化为可执行的物理结构。**Semantic World Models for Robots: Reducing Predictive Waste in Robotic Control** 认为像素级预测会浪费大量建模能力，并主张使用更语义化、更控制相关的预测目标来提升机器人世界模型效率。**CausalPhysics: Unifying Semantic Reasoning, Physical Dynamics, and Counterfactual Simulation in World Models** 尝试将语义推理、物理动态和反事实模拟结合起来，为因果世界模型提供了更接近本文 causal belief ledger 的思想背景。总体来看，已有工作已经从密集视频未来预测逐步发展到高效 WAM、潜空间 WAM、对象寻址 WAM、层级规划 WAM、自适应执行 WAM 和结构化评测体系，但这些方法大多仍缺少对“计划依赖了哪些未经验证的因果假设”的显式建模；因此，本文提出 Ledger\-WAM，通过 causal belief ledger、causal debt 和 self\-healing planner，将 WAM 的长时程想象过程转化为可追踪、可验证、可回滚的因果信念维护过程。

# 提出的方法



## 3\.1 Overview

我们研究语言条件下的长时程机器人操作任务。给定历史观测、机器人本体状态、历史动作和语言指令，目标是在部分可观测环境中生成一系列动作，使机器人完成多阶段操作任务。我们将历史信息记为 \(h\_t=\(o\_\{\\leq t\}, q\_\{\\leq t\}, a\_\{\<t\}\)\)，其中 \(o\_t\) 表示当前视觉观测，\(q\_t\) 表示机器人本体状态，\(a\_t\) 表示动作，\(g\) 表示语言任务指令。传统 World Action Model 通常学习如下形式的动作条件未来预测和动作生成：

\[

p\_\\theta\(a\_t, \\hat\{o\}\_\{t\+1:t\+H\} \\mid h\_t, g\),

\]

其中 \(\\hat\{o\}\_\{t\+1:t\+H\}\) 是模型想象出的未来观测序列。然而，在长时程任务中，显式未来 rollout 容易在接触、遮挡、抓取稳定性和物体持续性等关键状态上产生累积误差。本文的核心观点是，长时程失败并不只是未来图像预测不准确，而是模型在执行过程中不断依赖未经验证的因果假设。因此，我们将 WAM 的未来想象从密集观测预测转化为因果信念维护。

我们提出 **Ledger\-WAM**，其核心状态不是单一 latent state，也不是一段未来视频，而是一个随时间更新的因果信念账本：

\[

\\mathcal\{L\}\_t = \{c\_t^1, c\_t^2, \\ldots, c\_t^\{N\_t\}\},

\]

其中每个 \(c\_t^i\) 表示当前任务规划所依赖的一条动作条件因果声明。Ledger\-WAM 的整体建模形式为：

\[

z\_t = E\_\\theta\(h\_t, g\),

\]

\[
\\mathcal\{L\}*t = U*\\theta\(\\mathcal\{L\}*\{t\-1\}, z\_t, a*\{t\-1\}, o\_t\),
\]

\[

a\_t = \\pi\_\\theta\(z\_t, \\mathcal\{L\}\_t, g\)\.

\]

其中 \(E\_\\theta\) 是多模态状态编码器，\(U\_\\theta\) 是因果信念账本更新器，\(\\pi\_\\theta\) 是基于账本的动作策略。与传统 WAM 相比，Ledger\-WAM 不仅预测下一步动作，还显式记录当前计划依赖了哪些世界事实、这些事实是否被观测验证、如果这些事实错误会造成多大风险，以及一旦错误应该回滚到哪个执行阶段。

具体而言，Ledger\-WAM 包含三个主要组件。第一，多模态状态编码器将视觉观测、语言指令、机器人状态和历史动作编码为任务相关的对象级表示。第二，因果信念账本模块生成并更新动作条件因果声明，记录对象关系、动作前提、动作效果、观测证据、置信度、因果债务和回滚位置。第三，自修复规划器根据账本状态在任务动作和修复动作之间进行选择。当账本中的关键因果声明可信时，模型继续执行任务动作；当某些声明具有较高因果债务时，模型优先执行验证、修复或局部回滚动作。

整个执行过程可以表示为一个闭环：

\[
h\_t \\rightarrow z\_t \\rightarrow \\mathcal\{L\}*t \\rightarrow a\_t \\rightarrow o*\{t\+1\} \\rightarrow \\mathcal\{L\}\_\{t\+1\}\.
\]

该闭环使 WAM 不再一次性依赖长程 rollout，而是在每一步真实执行后更新其因果信念，并根据新的证据决定继续推进任务、主动验证关键假设，或回滚到最近可靠状态。

## 3\.2 因果信念账本

Ledger\-WAM 的核心表示是因果信念账本。每条因果声明 \(c\_t^i\) 被定义为一个结构化元组：

\[

c\_t^i = \(e\_i, r\_i, p\_i, u\_i, b\_t^i, s\_t^i, d\_t^i, \\tau\_t^i\),

\]

其中 \(e\_i\) 表示该声明涉及的实体集合，例如目标物体、容器、工具或机器人末端执行器；\(r\_i\) 表示实体之间的关系，例如接触、支撑、包含、遮挡、相对位置或同步运动；\(p\_i\) 表示该声明成立所需的动作前提；\(u\_i\) 表示该声明对后续世界状态的动作效果；\(b\_t^i\) 表示当前观测证据；\(s\_t^i\) 表示该声明的可信度；\(d\_t^i\) 表示该声明的因果债务；\(\\tau\_t^i\) 表示该声明失效时应回滚到的历史阶段。

例如，在“抓起杯子并放入盒子”的任务中，模型可能生成如下声明：杯子与夹爪处于接触状态，杯子受到夹爪支撑，夹爪闭合后杯子应随夹爪运动，杯子虽然被遮挡但仍位于夹爪附近。这些声明共同构成后续移动和放置动作的前提。若其中某条声明缺乏证据或被真实观测推翻，后续计划就不应继续无条件依赖它。

给定编码后的状态 \(z\_t\)，账本生成器首先预测候选因果声明：

\[
\\tilde\{\\mathcal\{L\}\}*t = G*\\theta\(z\_t, g\),
\]

然后结合上一时刻账本和新观测进行增量更新：

\[
\\mathcal\{L\}*t = U*\\theta\(\\mathcal\{L\}\_\{t\-1\}, \\tilde\{\\mathcal\{L\}\}*t, a*\{t\-1\}, o\_t\)\.
\]

对于每条声明，模型根据当前观测证据更新其可信度：

\[

s\_t^i = \\sigma \\big\( f\_\\theta\(z\_t, c\_t^i, b\_t^i\) \\big\),

\]

其中 \(\\sigma\) 是 sigmoid 函数，\(f\_\\theta\) 是声明可信度估计网络。若新观测支持该声明，例如物体随夹爪运动，则 \(s\_t^i\) 上升；若新观测与该声明冲突，例如夹爪移动而物体停留在桌面，则 \(s\_t^i\) 下降。

为了刻画继续依赖某条声明的风险，我们为每条声明定义因果债务 \(d\_t^i\)。因果债务不仅取决于声明本身的不确定性，也取决于它对后续任务的影响、可观测性、可修复性和失败代价。具体地，我们将因果债务建模为：

\[

d\_t^i =

\\sigma \\big\(

w\_1\(1\-s\_t^i\)

- w\_2 u\_t^i

- w\_3 \\eta\_t^i

- w\_4 \\rho\_t^i

- w\_5\(1\-\\omega\_t^i\)
\\big\),
\]

其中 \(u\_t^i\) 表示模型对该声明的预测不确定性，\(\\eta\_t^i\) 表示后续动作对该声明的依赖程度，\(\\rho\_t^i\) 表示该声明错误后的修复代价，\(\\omega\_t^i\) 表示该声明当前是否容易被观测验证。直观上，如果某条声明置信度低、后续动作高度依赖它、错误后难以恢复，并且当前又难以直接观测，那么它的因果债务就会很高。

后续动作对声明的依赖程度由任务计划中的动作依赖预测器给出：

\[

\\eta\_t^i =

\\frac\{1\}\{K\}

\\sum\_\{k=1\}^\{K\}

D\_\\theta\(c\_t^i, \\hat\{a\}\_\{t\+k\}, g\),

\]

其中 \(\\hat\{a\}*\{t\+k\}\) 是模型预测的未来候选动作，\(D*\\theta\) 用于判断某条声明是否是未来动作执行的必要前提。例如，“目标物体已经稳定处于夹爪中”会被移动、放置、倾倒等多个后续动作依赖，因此其 \(\\eta\_t^i\) 较高。

整个账本的全局风险定义为所有声明因果债务的加权和：

\[
D\(\\mathcal\{L\}*t\) =*
*\\sum*\{i=1\}^\{N\_t\}
\\alpha\_t^i d\_t^i,
\]

其中 \(\\alpha\_t^i\) 表示声明 \(c\_t^i\) 对当前任务目标的重要性。这样，模型可以同时知道单个声明的风险和当前账本整体是否可靠。

Ledger\-WAM 的训练目标由动作预测、声明预测、因果债务校准、回滚预测和反事实一致性组成：

\[
\\mathcal\{J\}\(\\theta\)

\\mathcal\{L\}*\{act\}*
*\+*
*\\lambda\_1 \\mathcal\{L\}*\{claim\}
\+
\\lambda\_2 \\mathcal\{L\}*\{debt\}*
*\+*
*\\lambda\_3 \\mathcal\{L\}*\{rollback\}
\+
\\lambda\_4 \\mathcal\{L\}\_\{cf\}\.
\]

其中动作预测损失为：

\[
\\mathcal\{L\}\_\{act\}

\-\\log \\pi\_\\theta\(a\_t^\\ast \\mid z\_t, \\mathcal\{L\}\_t, g\),

\]

其中 \(a\_t^\\ast\) 是专家动作。声明预测损失用于监督模型判断某个因果声明是否真实成立：

\[
\\mathcal\{L\}\_\{claim\}

\-\\sum\_\{i=1\}^\{N\_t\}

\\left\[

y\_t^i \\log s\_t^i

\+

\(1\-y\_t^i\)\\log\(1\-s\_t^i\)

\\right\],

\]

其中 \(y\_t^i\) 是来自模拟器状态、事件解析器或人工标注的声明真值。因果债务校准损失用于使预测债务与真实失败风险对齐：

\[
\\mathcal\{L\}\_\{debt\}

\\sum\_\{i=1\}^\{N\_t\}

\\left\|

d\_t^i \- \\bar\{d\}\_t^i

\\right\|,

\]

其中 \(\\bar\{d\}\_t^i\) 表示由失败轨迹、修复轨迹或任务依赖图得到的目标债务值。回滚预测损失用于监督模型在声明失效时选择正确的回滚阶段：

\[
\\mathcal\{L\}\_\{rollback\}

\-\\sum\_\{i=1\}^\{N\_t\}

\\log p\_\\theta\(\\tau\_t^\{i\\ast\} \\mid c\_t^i, \\mathcal\{L\}\_t, h\_t\),

\]

其中 \(\\tau\_t^\{i\\ast\}\) 是真实或离线推断得到的回滚位置。为了增强账本的动作条件因果性，我们还引入反事实一致性损失。对于同一状态下的两个不同动作 \(a\_t\) 和 \(a'\_t\)，模型应预测不同的账本变化：

\[
\\mathcal\{L\}\_\{cf\}

\\max
\\left\(
0,
m \-
\\left\|
\\Delta \\mathcal\{L\}\_t\(a\_t\)

\\Delta \\mathcal\{L\}\_t\(a'\_t\)

\\right\|\_1

\\right\),

\]

其中 \(\\Delta \\mathcal\{L\}\_t\(a\_t\)\) 表示执行动作 \(a\_t\) 后账本的预测变化，\(m\) 是间隔超参数。该损失鼓励模型学习动作如何改变对象关系和因果状态，而不是仅仅学习视觉共现模式。

## 3\.3 自修复规划器

基于因果信念账本，Ledger\-WAM 在执行阶段引入自修复规划器，使模型能够在任务动作和修复动作之间动态选择。我们将动作空间划分为两部分：

\[
\\mathcal\{A\} = \\mathcal\{A\}*\{task\} \\cup \\mathcal\{A\}*\{repair\},
\]

其中 \(\\mathcal\{A\}*\{task\}\) 包含直接推进任务目标的动作，例如抓取、移动、放置、打开、关闭、插入和推动；\(\\mathcal\{A\}*\{repair\}\) 包含用于验证或修正因果声明的动作，例如轻微抬升、调整视角、短距离回撤、重新闭合夹爪、触觉检查、重新对齐和局部重抓。

在每个时刻，规划器首先根据当前账本计算高风险声明集合：

\[

\\mathcal\{H\}\_t =

\{c\_t^i \\mid d\_t^i \> \\delta,\\ \\alpha\_t^i \> \\epsilon \},

\]

其中 \(\\delta\) 是债务阈值，\(\\epsilon\) 是任务重要性阈值。若不存在高风险关键声明，模型执行正常任务动作：

\[
a\_t^\{task\}

\\arg\\max\_\{a \\in \\mathcal\{A\}*\{task\}\}*
*\\pi*\\theta\(a \\mid z\_t, \\mathcal\{L\}\_t, g\)\.
\]

若存在高风险关键声明，模型不立即推进任务，而是评估候选修复动作对账本风险的降低效果。对于任意候选修复动作 \(a\)，模型先预测执行该动作后的未来观测和账本状态：

\[
\\hat\{o\}*\{t\+1\}, \\hat\{\\mathcal\{L\}\}*\{t\+1\}

W\_\\theta\(o\_t, \\mathcal\{L\}\_t, a, g\),

\]

其中 \(W\_\\theta\) 是轻量级动作条件世界预测器。候选修复动作的价值由三部分组成：预期债务下降、动作代价和任务风险：

\[
S\(a\)

\\mathbb\{E\}
\\left\[
D\(\\mathcal\{L\}\_t\)

D\(\\hat\{\\mathcal\{L\}\}\_\{t\+1\}\)
\\right\]

\\beta C\(a\)

\\gamma R\(a\),

\]

其中 \(C\(a\)\) 表示动作执行代价，\(R\(a\)\) 表示该动作破坏当前任务状态的风险。修复动作选择为：

\[
a\_t^\{repair\}

\\arg\\max\_\{a \\in \\mathcal\{A\}\_\{repair\}\}

S\(a\)\.

\]

最终动作选择规则为：

\[

a\_t =

\\begin\{cases\}

a\_t^\{task\}, \& D\(\\mathcal\{L\}\_t\) \< \\tau, \\

a\_t^\{repair\}, \& D\(\\mathcal\{L\}\_t\) \\geq \\tau,

\\end\{cases\}

\]

其中 \(\\tau\) 是全局账本风险阈值。该规则使模型在账本可靠时高效推进任务，在关键假设不可靠时优先降低因果债务。

执行修复动作后，模型根据真实观测更新账本：

\[
\\mathcal\{L\}\_\{t\+1\}

U\_\\theta\(\\mathcal\{L\}*t, z*\{t\+1\}, a\_t, o\_\{t\+1\}\)\.
\]

如果某条高风险声明被验证，则其可信度上升、因果债务下降，规划器继续执行任务动作；如果该声明被推翻，则模型触发局部回滚。我们用冲突分数判断声明是否失效：

\[
v\_\{t\+1\}^i

\\mathbb\{I\}

\\left\[

s\_\{t\+1\}^i \< \\kappa

\\ \\text\{and\}

d\_\{t\+1\}^i \> \\delta

\\right\],

\]

其中 \(\\kappa\) 是可信度阈值。若 \(v\_\{t\+1\}^i=1\)，模型将账本和任务状态回滚到该声明对应的回滚位置：

\[
\\mathcal\{L\}*\{t\+1\}*
*\\leftarrow*
*\\text\{Rollback\}\(\\mathcal\{L\}*\{t\+1\}, \\tau\_t^i\)\.
\]

同时，任务规划器只重新生成与该声明相关的局部子计划，而不是重启整个长时程任务：

\[

a\_\{t\+1:t\+K\}

\\sim

\\pi\_\\theta

\(\\cdot \\mid z\_\{t\+1\}, \\text\{Rollback\}\(\\mathcal\{L\}\_\{t\+1\}, \\tau\_t^i\), g\)\.

\]

例如，如果“目标物体已经被稳定抓住”的声明被推翻，模型只回滚到抓取阶段并生成重新抓取动作；如果“目标物体已经放入容器”的声明被推翻，模型只回滚到放置阶段，而不重新执行搜索和打开容器等早期子任务。

自修复规划器的训练目标包括修复动作选择和回滚决策两部分。给定专家或自动生成的修复动作 \(a\_t^\{repair\\ast\}\)，修复策略损失为：

\[
\\mathcal\{L\}\_\{repair\}

\-\\log

\\pi\_\\theta^\{repair\}

\(a\_t^\{repair\\ast\} \\mid z\_t, \\mathcal\{L\}\_t, g\)\.

\]

同时，为了鼓励修复动作真正降低账本风险，我们引入债务下降奖励：

\[
r\_t^\{repair\}

D\(\\mathcal\{L\}\_t\)

D\(\\mathcal\{L\}\_\{t\+1\}\)

\\beta C\(a\_t\)

\\gamma R\(a\_t\)\.

\]

最终规划器训练目标为：

\[
\\mathcal\{L\}\_\{planner\}

\\mathcal\{L\}*\{act\}*
*\+*
*\\mu\_1 \\mathcal\{L\}*\{repair\}

\\mu\_2 r\_t^\{repair\}

\+

\\mu\_3 \\mathcal\{L\}\_\{rollback\}\.

\]

通过上述机制，Ledger\-WAM 在执行过程中形成“任务推进—因果验证—局部修复—回滚恢复”的闭环。与传统 WAM 的一次性未来 rollout 或整体重规划不同，Ledger\-WAM 能够定位具体高风险因果声明，并通过最小代价修复动作恢复可靠信念，从而提升长时程操作任务中的成功率、失败定位能力和执行效率。

# 实验

## 数据集介绍

**RoboTwin 2\.0 / RMBench\.** RoboTwin 2\.0 / RMBench 是本文用于评估复杂长时程操作能力的主要 benchmark。RMBench 构建在 RoboTwin 2\.0 平台之上，重点面向 memory\-dependent robotic manipulation，包含多种需要跨时间记忆、状态追踪和动作依赖推理的操作任务。该数据集特别适合评估 Ledger\-WAM 的因果信念维护能力，因为其中许多任务要求机器人在执行过程中记住先前观测到的物体位置、操作阶段或隐含状态，并在后续动作中正确利用这些信息。对于本文方法而言，RoboTwin 2\.0 / RMBench 可以用于检验 causal belief ledger 是否能够追踪长时程任务中的关键状态假设，例如目标物体是否仍在预期位置、物体是否已经被成功抓取、某个中间状态是否已经完成，以及 self\-healing planner 是否能够在记忆不确定或状态被遮挡时主动执行验证和修复动作。

**LIBERO\-Long\.** LIBERO\-Long 是本文用于标准长时程语言条件机器人操作评测的 benchmark。LIBERO 原本面向 lifelong robot learning，包含多个不同类型的语言条件操作任务集合，其中 LIBERO\-Long 专门关注由多个子目标组成的长时程任务，常用于评估机器人策略在语言理解、子任务组合、任务顺序保持和长程执行稳定性方面的能力。该数据集具有较高的社区认可度，适合作为与现有 VLA、diffusion policy、hierarchical policy 和 WAM 方法进行公平比较的标准实验环境。对于 Ledger\-WAM 而言，LIBERO\-Long 可以验证方法是否能够在标准语言条件操作任务中减少误差累积，是否能够通过因果债务识别关键中间状态的风险，并在抓取、移动、放置、开关容器等多阶段任务中提升整体成功率和执行稳定性。

**VLABench\.** VLABench 是本文用于评估语义理解、组合泛化和长时程推理能力的大规模语言条件机器人操作 benchmark。该数据集包含大量任务类别和物体实例，强调自然语言指令、隐式人类意图、空间关系、物理规律、常识迁移和多步推理等能力，因此比传统操作 benchmark 更适合评估具身模型在开放语义场景中的泛化能力。VLABench 对 Ledger\-WAM 尤其重要，因为本文提出的 causal belief ledger 不仅需要追踪低层接触和抓取状态，还需要将语言指令中的对象关系、目标约束和子目标依赖转化为可维护的因果声明。例如，模型需要明确“目标物体是哪一个”“它与其他物体的空间关系是什么”“当前子目标是否已经完成”以及“后续动作依赖哪些尚未验证的语义条件”。因此，VLABench 可以用于检验 Ledger\-WAM 是否具备从语言语义到因果信念的建模能力，并进一步验证其在组合任务和复杂指令下的泛化性能。

## 实现细节

## 评价指标

## 实验结果和分析

## 消融实验

# conclusion

## 工作总结

## 贡献总结

## 未来工作

