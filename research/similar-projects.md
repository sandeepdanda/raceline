# Similar Projects & X-Factors to Adopt into Raceline

**Date:** 2026-06-29 PST
**Author:** research pass via WebFetch on real GitHub repos + live demos
**Raceline (my project):** 2D top-down car with distance-ray sensors learns to lap a track via PPO (Stable-Baselines3 + custom Gymnasium env), policy exported to ONNX and run live in the browser (onnxruntime-web, canvas demo with trained-vs-untrained toggle + sensor rays drawn). Python 3.11, CPU training, ~4 min.
**Confidence:** High on repos fetched directly (star counts quoted from live pages on 2026-06-29). Medium where a demo's canvas exposed no DOM (overlay specifics inferred), flagged inline.

---

## TL;DR

My ONNX-in-browser angle is already rare in this space and is the differentiator. The viral DNA of every starred project is the same three things: **many agents on screen at once**, a **live counter that visibly climbs**, and **one-click interactivity** (mutate / save brain / change population / draw a track). Genetic-algorithm demos win the visual drama because they show the whole population dying in parallel; RL demos usually show one finished agent. The good news: I can fake 90% of that drama cheaply with one trained ONNX policy by spawning N noisy copies + a clean "ghost" car, without lying about the fact that PPO (not evolution) trained it.

The single highest wow-per-effort wins for Raceline, in order: **(1) N ghost cars from one policy, (2) draw-your-own-track, (3) live network-activation overlay, (4) save/load + share a run, (5) racing-line ghost + reward shaping.**

---

## Notable projects and their x-factors

### Cluster A - RL self-driving / racing (closest to my method)

| Project | Stars (2026-06-29) | X-factor(s) | Adopt? |
|---|---|---|---|
| [tmrl](https://github.com/trackmania-rl/tmrl) | 720 | **"19-beam LIDAR" observation mode** - literally my distance-ray sensor design, with a real-world precedent. Real-time elastic time-stepping (rtgym). | High (ray framing validates mine); Low for the distributed/real-time parts |
| [Linesight-RL/linesight](https://github.com/Linesight-RL/linesight) | 739 | **RL that beat Trackmania world records** on official tracks (May 2024). Aspirational narrative. | Low to reuse (game-coupled); High as framing/copy |
| [xtma/pytorch_car_caring](https://github.com/xtma/pytorch_car_caring) | 159 | **Beta-distribution action head** (bounded, stable continuous steering/throttle); **frame-stack + action-repeat to encode velocity**. | High - SB3 supports custom distributions; drop-in |
| [andywu0913/OpenAI-GYM-CarRacing-DQN](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) | 96 | **Human-drive vs model-drive** scripts + overfitting narrative (500-ep model smooth, 600-ep reckless). | High - adding a "you drive" toggle is small |
| [elsheikh21/car-racing-ppo](https://github.com/elsheikh21/car-racing-ppo) | 52 | Clean frame-preprocessing pipeline + demo GIF. | High - standard preprocessing |
| [dgnzlz/Capstone_AWS_DeepRacer](https://github.com/dgnzlz/Capstone_AWS_DeepRacer) | 195 | **K1999 racing-line reward shaping** (12th of 1291 in F1 time-trial); K-Means action-space tuning. | High for the reward-shaping concept |
| [deepracer-analysis](https://github.com/aws-deepracer-community/deepracer-analysis) | 174 | **Post-hoc log-analysis notebooks** turning training logs into visual diagnostics. | Med - build equivalents from SB3 Monitor CSV |
| [deepracer-for-cloud](https://github.com/aws-deepracer-community/deepracer-for-cloud) | 352 | **MP4 export of evaluation runs.** | Low to adopt stack; Med for the MP4-of-a-lap idea |
| [TUMFTM/global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) | 588 | **Minimum-curvature optimal racing line** generator (standalone Python). | Med - precompute a target line, draw as overlay / use in reward |

### Cluster B - Watch-it-learn browser demos (closest to my UX)

| Project | Stars (2026-06-29) | X-factor(s) | Adopt? |
|---|---|---|---|
| [gniziemazity/Self-driving-car](https://github.com/gniziemazity/Self-driving-car) (Radu, "no libraries") | 755 | **The template.** Many parallel cars sharing one best brain with **sensor rays drawn on every car**; **save 💾 / discard 🗑️ brain** buttons + **car-count selector (1/10/100/1000)** + mutation slider. Live: [radufromfinland.com](https://radufromfinland.com/projects/selfdrivingcar/) | High - maps directly onto my stack |
| [gniziemazity/virtual-world](https://github.com/gniziemazity/virtual-world) | 232 | **OSM road editor + minimap** (Phase 2 follow-up) - draw/import a world. | Med - inspiration for draw-your-track |
| [xviniette/FlappyLearning](https://github.com/xviniette/FlappyLearning) | ~4k | **Dozens of birds at once** + live generation/max-score counter; **speed-up control**. | Med - many-agents High; 50 ONNX passes/frame is the cost |
| [ssusnic/Machine-Learning-Flappy-Bird](https://github.com/ssusnic/Machine-Learning-Flappy-Bird) | ~1.8k | **On-screen status overlay** (`drawStatus`) showing what each agent senses; tiny named input set. | High - "what the car senses" HUD is cheap, high-impact |
| MarI/O (SethBling) - [video](https://www.youtube.com/watch?v=qv6UVOQ0F44) | n/a (video) | **Live neural-network graph drawn over gameplay** - nodes + connections light up as the agent acts; corner HUD with Generation/Fitness. *(overlay specifics: general knowledge, Medium confidence)* | High and recommended - my PPO net is a fixed MLP, easy to draw + animate |
| [victorqribeiro/aimAndShoot](https://github.com/victorqribeiro/aimAndShoot) | 234 | Per-agent floating health/cooldown bars; **player-in-the-loop** twist. | Med - status bars easy; player-interferes is a bigger redesign |
| [CodingTrain/Toy-Neural-Network-JS](https://github.com/CodingTrain/Toy-Neural-Network-JS) | 437 | Roadmap literally names "many players at once" + "steering sensors" - confirms parallel-agent rendering is the recognized core appeal. | High as idea source |

### Cluster C - Genetic-algorithm / neuroevolution car demos

| Project | Stars (2026-06-29) | X-factor(s) | Adopt? |
|---|---|---|---|
| [ArztSamuel/Applying_EANNs](https://github.com/ArztSamuel/Applying_EANNs) | 1,573 | **The canonical "AI learns to drive" look:** whole population drives simultaneously, **current-best car's data always displayed**, **generation counter** lower-left. 5 distance rays -> fixed net, fitness = % course complete (1:1 with my env). | Med - reimplement (Unity/C#), but model maps directly |
| [trekhleb/self-parking-car-evolution](https://github.com/trekhleb/self-parking-car-evolution) | 759 | **Cleanest browser GA reference** - live in-browser evolution, on-screen generation/genome controls, polished GA explainer page. TypeScript + Three.js. Live: [trekhleb.dev](https://trekhleb.dev/self-parking-car-evolution) | Med-High - same canvas territory; copy the UX + explainer structure |
| [Code-Bullet/Smart-Dots](https://github.com/Code-Bullet/Smart-Dots-Genetic-Algorithm-Tutorial) | 187 | Archetypal **whole-population-released-at-once + generation counter + fittest survives** UX. | Med as UX template (Processing, not reusable code) |
| [jhanreg11/NeuroCars](https://github.com/jhanreg11/NeuroCars) | 3 | **Closest small repo to my exact model:** 5 distance-ray sensors, fitness = time alive + distance. Live: [jacob-hanson.com/neurocars](https://www.jacob-hanson.com/neurocars) | High as a direct reference |
| [harishkotra/NeuroDrive](https://github.com/harishkotra/NeuroDrive) | 1 | **Procedurally generated tracks** - population learns on a fresh track each time. | Med - procedural-track idea |
| BoxCar2D (boxcar2d.com) | n/a | Evolves the **physical shape** of a side-view vehicle over Box2D terrain; "best so far" ghost. *(site unreachable 2026-06-29, Low confidence)* | Low - morphology evolution doesn't fit fixed-car PPO; skip |

**Why GA demos look cooler than RL demos:** they show the whole population (parallel-death spectacle), a generation counter that ticks up, and the best-of-generation highlighted with live-climbing fitness. RL/PPO trains in vectorized envs off-screen and only shows the final agent. My honest RL analog to "generation 1 vs N" is the **trained-vs-untrained toggle I already have** - keep it front and center.

---

## Ranked x-factors to adopt into Raceline (by wow / effort)

Effort: **S** = a day or less, **M** = a few days, **L** = a week+.

### 1. N ghost cars from one policy - S - **do this first**
- **What:** Spawn 10-100 cars all running the *same* trained ONNX policy; diverge them with jittered spawn positions/headings + small Gaussian noise on actions or sensor inputs. They fan out, some crash, the clean ones survive.
- **Why wow:** This is the single most screenshot-shared image in the entire genre (Radu, FlappyLearning, Applying_EANNs). It manufactures the "population dying in parallel" drama that makes GA demos go viral - with zero retraining.
- **Reference:** [gniziemazity/Self-driving-car](https://github.com/gniziemazity/Self-driving-car) (car-count 1/10/100/1000), [ArztSamuel/Applying_EANNs](https://github.com/ArztSamuel/Applying_EANNs).
- **Watch out:** N forward passes/frame in onnxruntime-web. At ~100 cars keep the net tiny (you already do) and batch the inference into one ONNX call per frame.

### 2. Draw-your-own-track - M - **biggest genuine differentiator**
- **What:** Canvas spline editor -> feed into my existing ray-sensor env -> run the ONNX policy live on the user's track. None of the famous demos let you *draw* a track and watch a pretrained policy attempt it.
- **Why wow:** Turns the demo from "watch a replay" into "test my AI on YOUR track." Highest stickiness. Generalization will be imperfect (policy trained on one track) - which is itself an honest, interesting thing to surface ("watch it struggle on a track it's never seen").
- **Reference:** procedural tracks in [harishkotra/NeuroDrive](https://github.com/harishkotra/NeuroDrive); OSM editor in [gniziemazity/virtual-world](https://github.com/gniziemazity/virtual-world). The *draw-your-own* twist is unclaimed - genuine novelty.

### 3. Live network-activation overlay - M - **highest "premium" visual**
- **What:** Draw my PPO MLP once (nodes = ray inputs -> hidden -> steer/throttle outputs); animate edge color/width by live activation each frame from onnxruntime-web.
- **Why wow:** The defining "watch it think" moment of the genre (MarI/O). My policy is a fixed MLP so this is far easier than for a CNN.
- **Reference:** MarI/O ([video](https://www.youtube.com/watch?v=qv6UVOQ0F44)). *(overlay specifics Medium confidence - general knowledge.)*
- **Watch out:** onnxruntime-web may not expose intermediate activations directly; you may need to export hidden-layer outputs as extra ONNX graph outputs, or re-derive the forward pass in JS.

### 4. Save / load / share a run - S - **the "I trained this" loop**
- **What:** Button to download/upload the ONNX policy (or a seed), and a "share this run" link (encode track + seed in the URL).
- **Why wow:** Radu's save/discard-brain buttons are precisely what turns viewers into participants and earns stars. Shareable runs are the growth loop.
- **Reference:** [gniziemazity/Self-driving-car](https://github.com/gniziemazity/Self-driving-car) save 💾/discard 🗑️ brain.

### 5. "What the car senses" mini-HUD - S
- **What:** Render the live ray distances as a row of bars + the raw action outputs (steer/throttle) as gauges.
- **Why wow:** Makes the policy legible - the viewer understands *why* it turns. Cheap, high teaching value.
- **Reference:** [ssusnic/Machine-Learning-Flappy-Bird](https://github.com/ssusnic/Machine-Learning-Flappy-Bird) `drawStatus`.

### 6. Racing-line ghost + reward shaping - M - **makes it lap FAST not just finish**
- **What:** Precompute a minimum-curvature optimal line for the track offline; (a) draw it as a translucent ghost the car chases, and (b) add distance-to-line as a reward shaping term so the policy laps fast, not just completes.
- **Why wow:** Elevates from "doesn't crash" to "drives a clean racing line." Ghost-lap is a familiar racing-game trope.
- **Reference:** [TUMFTM/global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) (min-curvature line), [dgnzlz/Capstone_AWS_DeepRacer](https://github.com/dgnzlz/Capstone_AWS_DeepRacer) (K1999 racing-line reward).

### 7. Parameter sliders (sensor count, speed, noise) - S
- **What:** Live sliders for sim speed (2x/4x), injected action noise, and number of ghost cars. A sensor-count slider needs retrain (fixed obs space), so expose it as a preset selector across a few pretrained policies instead.
- **Why wow:** The evolution-speed slider is idiomatic in GA demos (Coding Train); giving the viewer knobs increases dwell time.
- **Reference:** [CodingTrain/Toy-Neural-Network-JS](https://github.com/CodingTrain/Toy-Neural-Network-JS), FlappyLearning speed control.

### 8. Human-drive toggle - S
- **What:** Let the visitor drive with arrow keys on the same track, alongside the AI ghost.
- **Why wow:** "Can you beat the AI?" is instant engagement; pairs naturally with a leaderboard.
- **Reference:** [andywu0913/OpenAI-GYM-CarRacing-DQN](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN) (keyboard vs model scripts).

### 9. Leaderboard / lap-time ghost - M
- **What:** Record best lap times (yours + AI's) per track; replay the best as a translucent ghost car.
- **Why wow:** Ghost laps + leaderboard add competition and replayability. Pairs with #2 (draw-your-track) and #8 (human drive).
- **Reference:** "best so far" ghost pattern (BoxCar2D, general racing-game convention).

### 10. Export a lap to MP4/GIF for the README - M
- **What:** Record a trained lap and export as MP4/GIF for the repo README and portfolio.
- **Why wow:** A motion clip in the README massively outperforms a static screenshot for stars/clicks.
- **Reference:** [deepracer-for-cloud](https://github.com/aws-deepracer-community/deepracer-for-cloud) MP4 export.

### Training-side polish (not browser UX, but cheap credibility)
- **Beta-distribution action head** - S - bounded, stable continuous control ([xtma](https://github.com/xtma/pytorch_car_caring)).
- **Frame-stack / action-repeat to encode velocity** - S - so the policy "sees" speed ([xtma](https://github.com/xtma/pytorch_car_caring), [elsheikh21](https://github.com/elsheikh21/car-racing-ppo)).
- **"What did it learn" diagnostic plots** from SB3 Monitor CSV - M - reward curve, lap-completion over time ([deepracer-analysis](https://github.com/aws-deepracer-community/deepracer-analysis)).

---

## Opinionated recommendation

Ship in this order: **#1 (N ghost cars)** and **#5 (sensor HUD)** this week - both are S and together they instantly make the demo look like the viral GA ones. Then **#2 (draw-your-own-track)** as the headline feature that no famous demo has. Then **#3 (network overlay)** for the "wow, you can watch it think" screenshot. Keep the **trained-vs-untrained toggle** prominent throughout - it's the honest RL equivalent of "generation 1 vs N" and your project's existing edge.

Do **not** fake "generations" with noise-decay staging dressed up as evolution - it misrepresents that PPO trained the policy. The population + ghost + draw-your-track trio gets the GA-demo wow factor honestly. Lead the README copy with "PPO-trained in ~4 min on CPU, runs live in your browser" - the tiny-fast-model + no-server framing demonstrably drives stars (Radu's "no libraries", PufferLib's "super-human models in seconds").

---

## Unknowns / caveats

- Star counts for FlappyLearning (~4k) and ssusnic (~1.8k) are GitHub-rounded; treat as approximate. All counts observed 2026-06-29 and drift over time.
- MarI/O network-overlay specifics and aimAndShoot live-build controls are from general knowledge (canvas apps expose no DOM) - Medium confidence, flagged inline.
- BoxCar2D details are Low confidence - boxcar2d.com refused connection on 2026-06-29 and has no Wikipedia article.
- Yosh's Trackmania "AI learns to race" project has no canonical public repo found under guessed URLs - it appears to be a YouTube series only.
- onnxruntime-web exposing intermediate activations (#3) and batching N inferences/frame (#1) are the two technical risks to validate early.

## Sources

All observed 2026-06-29 via WebFetch / GitHub API.

1. https://github.com/trackmania-rl/tmrl (720)
2. https://github.com/Linesight-RL/linesight (739)
3. https://github.com/xtma/pytorch_car_caring (159)
4. https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN (96)
5. https://github.com/elsheikh21/car-racing-ppo (52)
6. https://github.com/aws-deepracer-community/deepracer-for-cloud (352)
7. https://github.com/aws-deepracer-community/deepracer-analysis (174)
8. https://github.com/dgnzlz/Capstone_AWS_DeepRacer (195)
9. https://github.com/TUMFTM/global_racetrajectory_optimization (588)
10. https://github.com/autorope/donkeycar (3.5k)
11. https://github.com/PufferAI/PufferLib (6.1k)
12. https://github.com/gniziemazity/Self-driving-car (755) + https://radufromfinland.com/projects/selfdrivingcar/
13. https://github.com/gniziemazity/virtual-world (232)
14. https://github.com/xviniette/FlappyLearning (~4k)
15. https://github.com/ssusnic/Machine-Learning-Flappy-Bird (~1.8k)
16. https://github.com/victorqribeiro/aimAndShoot (234) + https://victorqribeiro.github.io/aimAndShoot/
17. https://github.com/CodingTrain/Toy-Neural-Network-JS (437)
18. https://github.com/nature-of-code/noc-examples-p5.js (1.1k)
19. https://www.youtube.com/watch?v=qv6UVOQ0F44 (MarI/O)
20. https://github.com/ArztSamuel/Applying_EANNs (1,573) + https://youtu.be/rEDzUT3ymw4
21. https://github.com/trekhleb/self-parking-car-evolution (759) + https://trekhleb.dev/self-parking-car-evolution
22. https://github.com/Code-Bullet/Smart-Dots-Genetic-Algorithm-Tutorial (187)
23. https://github.com/jhanreg11/NeuroCars (3) + https://www.jacob-hanson.com/neurocars
24. https://github.com/harishkotra/NeuroDrive (1)
25. https://github.com/CodingTrain/website-archive (5,745)
26. boxcar2d.com (unreachable 2026-06-29)
