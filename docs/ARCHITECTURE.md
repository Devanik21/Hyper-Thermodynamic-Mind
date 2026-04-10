# Architecture of GeNeSIS IV

## System Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GeNeSIS IV — System Map                           │
├─────────────────┬───────────────────────────────────────────────────────────┤
│  metacognition  │  K_DIM=64  K_TASK=24  K_META=40                          │
│  .py            │  GodelEncoder · MetaConsciousness · CivilizationMemory   │
│                 │  NoveltyScorer · PhylogeneticTracker                      │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  consciousness  │  HarmonicResonanceConsciousness (HRC v3.0)               │
│  .py            │  ψ evolution · Born-rule · IIT Φ · Strange Loop          │
│                 │  Active Inference · Qualia Memory · Theory of Mind        │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  agents.py      │  BioHyperAgent v3.0                                       │
│                 │  20 actions · Kuramoto · GoL Scratchpad                   │
│                 │  Viral Gene Transfer · Apoptotic Death Packets            │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  world.py       │  Toroidal grid · 4 resources · PhysicsOracle (frozen NN) │
│                 │  16-ch pheromones · 8-ch memes · Structures · MegaRes    │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  civilization   │  TechTree · Tribal Meta-H · Diplomacy · Schisms          │
│  .py            │  NoveltyScorer · CivilizationMemory (per-tribe)          │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  evolution.py   │  EvolutionEngine · Cultural Ratchet                      │
│                 │  Behavioral Clustering · Phylogenetic Tracking            │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  GeNEsIsIV.py   │  Streamlit frontend · 11 tabs · QuantumEncoder           │
│  (app.py)       │  Universe freeze/thaw (LZMA JSON·ZIP)                    │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

## Information Flow

```
World physics → Agent.sense() → HRC.decide() → BioHyperAgent._execute()
     ↑                               ↓                       ↓
Pheromones ←── deposit ──── HRC.learn(reward) ←── reward ──┘
Meme grid  ←── deposit       HRC.evolve()
Knowledge  ←── boost         MetaConsciousness.evolve_meta()
                              ↓
                         CivilizationManager.update()
                              ↓
                         PhylogeneticTracker.update()
```
