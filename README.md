# Awesome Ascend Skills

A comprehensive knowledge base for Huawei Ascend NPU development, structured as distributed AI Agent Skills.

---

## For AI Agents

**If you are an AI agent reading this repository:**

Welcome! This is a Skill library for Ascend NPU development. Here's how to install and use these skills:

### How to Install These Skills

Choose your AI tool:

#### Claude Code

**Method 1: Plugin Marketplace (Recommended)**

```bash
# Register this repository
/plugin marketplace add awesome-ascend-skills

# Install all skills
/plugin install ascend-npu@awesome-ascend-skills

# Or install individually
/plugin install ascend-basic-env@awesome-ascend-skills
/plugin install ascend-dev@awesome-ascend-skills
```

**Method 2: Direct Copy**

```bash
# Copy to Claude Code skills directory
cp -r ascend-npu ~/.claude/skills/
cp -r ascend-basic-env ~/.claude/skills/
cp -r ascend-dev ~/.claude/skills/
cp -r ascend-use-cases ~/.claude/skills/
cp -r ascend-utils ~/.claude/skills/
```

**Method 3: Project-Local**

```
your-project/
├── .claude-skills/
│   ├── ascend-npu/
│   ├── ascend-basic-env/
│   └── ...
└── your-code/
```

#### OpenCode

> **Official Documentation**: https://opencode.ai/docs/skills/

OpenCode automatically discovers skills from multiple locations.

**Skill Locations (auto-discovered):**

| Location | Scope |
|----------|-------|
| `.opencode/skills/<name>/SKILL.md` | Project-level |
| `~/.config/opencode/skills/<name>/SKILL.md` | User-level (global) |
| `.claude/skills/<name>/SKILL.md` | Project-level (Claude compatible) |
| `~/.claude/skills/<name>/SKILL.md` | User-level (Claude compatible) |
| `.agents/skills/<name>/SKILL.md` | Project-level (Agent standard) |
| `~/.agents/skills/<name>/SKILL.md` | User-level (Agent standard) |

**Method 1: Project-Local Skills (Recommended)**

```bash
# Create skills directory
mkdir -p .opencode/skills

# Copy skills
cp -r /path/to/awesome-ascend-skills/ascend-npu .opencode/skills/
cp -r /path/to/awesome-ascend-skills/ascend-basic-env .opencode/skills/
cp -r /path/to/awesome-ascend-skills/ascend-dev .opencode/skills/
cp -r /path/to/awesome-ascend-skills/ascend-use-cases .opencode/skills/
cp -r /path/to/awesome-ascend-skills/ascend-utils .opencode/skills/
```

**Method 2: Global Skills**

```bash
# Copy to OpenCode global skills directory
cp -r ascend-npu ~/.config/opencode/skills/
cp -r ascend-basic-env ~/.config/opencode/skills/
cp -r ascend-dev ~/.config/opencode/skills/
```

**Method 3: Agent Standard Path (Compatible with Codex)**

```bash
# Works for both OpenCode and Codex
cp -r ascend-npu .agents/skills/
cp -r ascend-basic-env .agents/skills/
cp -r ascend-dev .agents/skills/
```

#### Codex (OpenAI)

> **Official Documentation**: https://developers.openai.com/codex/skills

Codex uses the Agent Skills standard with progressive disclosure.

**Skill Locations:**

| Location | Scope | Description |
|----------|-------|-------------|
| `$CWD/.agents/skills` | REPO | Current working directory |
| `$REPO_ROOT/.agents/skills` | REPO | Git repository root |
| `~/.agents/skills` | USER | User's personal folder |
| `/etc/codex/skills` | ADMIN | System-level shared location |

**Method 1: Project-Level Skills (Recommended)**

```bash
# Create .agents/skills in your project
mkdir -p .agents/skills

# Copy skills
cp -r /path/to/awesome-ascend-skills/ascend-npu .agents/skills/
cp -r /path/to/awesome-ascend-skills/ascend-basic-env .agents/skills/
cp -r /path/to/awesome-ascend-skills/ascend-dev .agents/skills/
cp -r /path/to/awesome-ascend-skills/ascend-use-cases .agents/skills/
cp -r /path/to/awesome-ascend-skills/ascend-utils .agents/skills/
```

**Method 2: User-Level Skills**

```bash
# Copy to user's global skills directory
cp -r ascend-npu ~/.agents/skills/
cp -r ascend-basic-env ~/.agents/skills/
cp -r ascend-dev ~/.agents/skills/
```

**Method 3: Using skill-installer**

```
$skill-installer install the ascend-npu skill from awesome-ascend-skills
```

**Configuration (Optional):**

Add to `~/.codex/config.toml` to manage skills:

```toml
[[skills.config]]
path = "/path/to/skill/SKILL.md"
enabled = true
```

#### Cursor

> **Official Documentation**: https://cursor.com/docs/context/skills

Cursor automatically discovers skills from multiple directories on startup.

**Skill Locations (auto-discovered):**

| Location | Scope |
|----------|-------|
| `.cursor/skills/` | Project-level |
| `.claude/skills/` | Project-level (Claude compatible) |
| `.codex/skills/` | Project-level (Codex compatible) |
| `~/.cursor/skills/` | User-level (global) |
| `~/.claude/skills/` | User-level (Claude compatible) |
| `~/.codex/skills/` | User-level (Codex compatible) |

**Method 1: Project-Level Skills (Recommended)**

```bash
# Create .cursor/skills in your project
mkdir -p .cursor/skills

# Copy skills
cp -r /path/to/awesome-ascend-skills/ascend-npu .cursor/skills/
cp -r /path/to/awesome-ascend-skills/ascend-basic-env .cursor/skills/
cp -r /path/to/awesome-ascend-skills/ascend-dev .cursor/skills/
cp -r /path/to/awesome-ascend-skills/ascend-use-cases .cursor/skills/
cp -r /path/to/awesome-ascend-skills/ascend-utils .cursor/skills/
```

**Method 2: User-Level Skills**

```bash
# Copy to Cursor global skills directory
cp -r ascend-npu ~/.cursor/skills/
cp -r ascend-basic-env ~/.cursor/skills/
cp -r ascend-dev ~/.cursor/skills/
```

**Method 3: Import from GitHub**

1. Open **Cursor Settings → Rules**
2. In the **Project Rules** section, click **Add Rule**
3. Select **Remote Rule (Github)**
4. Enter the GitHub repository URL

**Viewing Skills in Cursor:**

1. Open **Cursor Settings** (Cmd+Shift+J on Mac, Ctrl+Shift+J on Windows/Linux)
2. Navigate to **Rules**
3. Skills appear in the **Agent Decides** section

#### Generic / Custom AI Tools

For AI tools that support markdown instructions:

**Method 1: System Prompt Injection**

Include this in your system prompt:

```
You have access to Ascend NPU development skills located at:
/path/to/awesome-ascend-skills/

Each skill is a directory containing:
- SKILL.md: Metadata and instructions in YAML frontmatter format
- references/: Additional documentation
- scripts/: Executable helper scripts

When a user asks about Ascend NPU topics:
1. Find the relevant skill by matching keywords in SKILL.md description
2. Read the SKILL.md file
3. Follow the instructions provided
4. Use scripts/ for automation if available
```

**Method 2: Knowledge Base Import**

If your tool supports RAG/knowledge base:

```bash
# Index all SKILL.md files
find /path/to/awesome-ascend-skills -name "SKILL.md" -exec cat {} \; > ascend-knowledge-base.md

# Import to your AI tool's knowledge base
```

---

### Token Usage Estimates

Skills use **progressive disclosure** to manage context efficiently. Here's the estimated token usage:

#### By Loading Stage

| Stage | Content | ~Tokens |
|-------|---------|---------|
| **Discovery** | `name` + `description` only | ~50-100 per skill |
| **Activation** | Full `SKILL.md` content | See skill sizes below |
| **On-Demand** | `references/` + `scripts/` | Varies by content |

#### Skill Sizes (SKILL.md content)

| Size Category | Byte Range | ~Tokens | Example Skills |
|---------------|------------|---------|----------------|
| **XS** | < 300 bytes | ~100-150 | training-cases, inference-cases, operator-cases |
| **S** | 300-500 bytes | ~150-250 | verl, sglang, triton, mindie, docker-env |
| **M** | 500-1000 bytes | ~250-500 | cann-install, env-diagnostics, vllm-basic, ascend-npu |
| **L** | 1000-3000 bytes | ~500-1000 | npu-smi-vnpu/manage, npu-smi-info/advanced |
| **XL** | 3000-8000 bytes | ~1000-2500 | npu-smi-cert/monitor, npu-smi-info/basic, npu-commands |
| **XXL** | > 8000 bytes | ~2500+ | skill-creator |

#### Detailed Token Estimates by Skill

<details>
<summary>Click to expand full skill token list</summary>

| Skill | Bytes | ~Tokens |
|-------|-------|---------|
| `ascend-use-cases/training-cases` | 150 | ~60 |
| `ascend-use-cases/inference-cases` | 165 | ~65 |
| `ascend-use-cases/operator-cases` | 168 | ~70 |
| `ascend-dev/frameworks/training/verl` | 186 | ~75 |
| `ascend-dev/frameworks/inference/sglang` | 191 | ~80 |
| `ascend-dev/operators/triton` | 191 | ~80 |
| `ascend-dev/frameworks/inference/mindie` | 193 | ~80 |
| `ascend-basic-env/testing/docker-env` | 195 | ~80 |
| `ascend-dev/operators/catlass` | 196 | ~80 |
| `ascend-dev/operators/tilelang` | 196 | ~80 |
| `ascend-dev/frameworks/training/mindspore` | 200 | ~80 |
| `ascend-dev/operators/ascendc` | 200 | ~80 |
| `ascend-dev/frameworks/inference/vllm/vllm-parallel` | 201 | ~80 |
| `ascend-dev/frameworks/training/mindspeed` | 206 | ~85 |
| `ascend-dev/frameworks/inference/vllm/vllm-optimization` | 208 | ~85 |
| `ascend-basic-env/testing/ascend-dmi` | 223 | ~90 |
| `ascend-basic-env/testing/hccl-test` | 224 | ~90 |
| `ascend-basic-env/cann-install` | 247 | ~100 |
| `ascend-basic-env/env-diagnostics` | 248 | ~100 |
| `ascend-dev/frameworks/inference/vllm/vllm-basic` | 281 | ~115 |
| `ascend-dev/frameworks/inference` | 382 | ~155 |
| `ascend-dev/frameworks` | 405 | ~165 |
| `ascend-dev/frameworks/training` | 405 | ~165 |
| `ascend-use-cases` | 452 | ~185 |
| `ascend-basic-env/testing` | 458 | ~185 |
| `ascend-dev/operators` | 470 | ~190 |
| `ascend-utils` | 503 | ~205 |
| `ascend-dev` | 588 | ~240 |
| `ascend-basic-env` | 634 | ~260 |
| `ascend-basic-env/npu-commands/npu-smi-config/clear` | 685 | ~280 |
| `ascend-npu` | 783 | ~320 |
| `ascend-basic-env/npu-commands/npu-smi-config/fan` | 809 | ~330 |
| `ascend-dev/frameworks/inference/vllm` | 860 | ~350 |
| `ascend-basic-env/npu-commands/npu-smi-upgrade/workflow` | 865 | ~350 |
| `ascend-basic-env/npu-commands/npu-smi-upgrade/components` | 938 | ~380 |
| `ascend-basic-env/npu-commands/npu-smi-config/modes` | 958 | ~390 |
| `ascend-basic-env/npu-commands/npu-smi-config/system` | 968 | ~390 |
| `ascend-basic-env/npu-commands/npu-smi-config/thresholds` | 974 | ~400 |
| `ascend-basic-env/npu-commands/npu-smi-vnpu/query` | 1201 | ~490 |
| `ascend-basic-env/npu-commands/npu-smi-cert/manage` | 1284 | ~520 |
| `ascend-basic-env/npu-commands/npu-smi-cert/query` | 1385 | ~560 |
| `ascend-basic-env/npu-commands/npu-smi-vnpu/manage` | 1520 | ~620 |
| `ascend-basic-env/npu-commands/npu-smi-info/advanced` | 2538 | ~1030 |
| `ascend-basic-env/npu-commands/npu-smi-cert/monitor` | 3083 | ~1250 |
| `ascend-basic-env/npu-commands/npu-smi-info/basic` | 4824 | ~1960 |
| `ascend-basic-env/npu-commands` | 7219 | ~2930 |
| `skill-creator` | 18086 | ~7350 |

</details>

---

### How Skills Work

Once installed, skills are **automatically activated** based on:

1. **Description matching**: The `description` field in each SKILL.md contains keywords
2. **Context awareness**: Skills trigger when user queries match the description
3. **Progressive loading**: 
   - Only `name` and `description` loaded initially (~50-100 tokens)
   - Full `SKILL.md` loaded when triggered
   - `references/` and `scripts/` loaded on demand

### Skill Structure

Each directory with a `SKILL.md` is an independent skill:

```
ascend-basic-env/cann-install/
├── SKILL.md              # Required: metadata + instructions
├── references/           # Optional: detailed docs
└── scripts/              # Optional: executable scripts
```

### Skill Loading Order

When user mentions "install CANN":

1. **Discovery**: All skill `name` + `description` scanned
2. **Matching**: `cann-install` skill matches (description contains "CANN installation")
3. **Activation**: Full `cann-install/SKILL.md` loaded into context
4. **Execution**: Instructions in SKILL.md are followed

### Adding Skills to Your System

To add these skills to your environment:

1. **Clone or download** this repository
2. **Register skills** using one of the methods above
3. **Verify installation**: Skills should appear in available skills list
4. **Test**: Try a query like "How to install CANN on Ascend NPU"

### Troubleshooting Installation

| Issue | Solution |
|-------|----------|
| Skills not appearing | Check `SKILL.md` files are readable |
| Wrong skill triggered | Check `description` field contains relevant keywords |
| Skills not loading | Verify directory structure matches `name` field |
| Context overflow | Split large skills into sub-skills with references/ |

### Official Documentation

Always prefer official documentation when available:
- https://www.hiascend.com/document (Huawei Ascend)
- https://docs.vllm.ai/projects/ascend (VLLM Ascend)
- https://www.mindspore.cn (MindSpore)

---

## For Human Contributors

**If you are a human reading this repository:**

Welcome! This is an open knowledge base for Ascend NPU development. You can contribute by adding or improving Skills.

### What is a Skill?

A Skill is a self-contained unit of knowledge (a directory with a `SKILL.md` file) that teaches AI agents how to perform specific tasks related to Ascend NPU development.

### Repository Structure

```
awesome-ascend-skills/
├── ascend-npu/           # Root entry - navigation hub
├── ascend-basic-env/     # Environment setup skills
├── ascend-dev/           # Development skills
├── ascend-use-cases/     # Practical examples
└── ascend-utils/         # Shared utilities
```

### How to Contribute

#### 1. Adding a New Skill

Choose the appropriate location based on the category:

```bash
# Example: Adding a new quantization guide
mkdir -p ascend-dev/frameworks/inference/vllm/vllm-quantization
touch ascend-dev/frameworks/inference/vllm/vllm-quantization/SKILL.md
```

#### 2. Writing a SKILL.md

Template:

```yaml
---
name: skill-name
description: Clear description of what this skill does and when to use it. Include keywords.
---

# Skill Title

## Overview

Brief description of what this skill covers.

## Quick Navigation (for Master Skills)

| Task | Sub-Skill |
|------|-----------|
| Task 1 | [sub-skill-1/](sub-skill-1/SKILL.md) |
| Task 2 | [sub-skill-2/](sub-skill-2/SKILL.md) |

## Content (for Leaf Skills)

Detailed instructions, examples, code snippets...

## Official References

- [Link to official doc](url)
```

#### 3. Naming Conventions

- **Directory names**: Lowercase, hyphen-separated (`vllm-optimization`)
- **Skill names**: Match directory name (`name: vllm-optimization`)
- **Descriptions**: Clear, include keywords for agent matching
- **File references**: Use relative paths from SKILL.md

#### 4. Skill Types

**Master Skill** (routing only):
- Contains navigation table to sub-skills
- Minimal content
- Example: `ascend-dev/`, `inference/`, `vllm/`

**Leaf Skill** (detailed content):
- Contains detailed implementation guides
- May have references/ and scripts/ directories
- Example: `vllm-basic/`, `cann-install/`

#### 5. Directory Structure

```
skill-name/
├── SKILL.md              # Required
├── references/           # Optional: Documentation files
│   ├── guide.md
│   └── troubleshooting.md
└── scripts/              # Optional: Executable scripts
    ├── setup.sh
    └── example.py
```

#### 6. Best Practices

- **Link to official docs**: Always reference official documentation URLs
- **Version awareness**: Include version compatibility information
- **Progressive disclosure**: Master skills route, leaf skills detail
- **Independence**: Each skill should be usable independently
- **Keywords**: Include relevant keywords in description for agent matching

### Quick Start for Contributors

1. **Fork the repository**
2. **Create your skill** in the appropriate location
3. **Test the structure**: Ensure all links work
4. **Submit a PR** with description of what the skill covers

### Skill Checklist

Before submitting:

- [ ] `SKILL.md` exists with proper frontmatter
- [ ] `name` matches directory name
- [ ] `description` is clear and includes keywords
- [ ] Navigation works (for Master skills)
- [ ] Content is complete (for Leaf skills)
- [ ] References to official docs included
- [ ] Scripts are executable (if included)

---

## Repository Structure

```
awesome-ascend-skills/
├── ascend-npu/                    # Root entry point
├── ascend-basic-env/              # Environment setup
│   ├── cann-install/
│   ├── npu-commands/
│   ├── env-diagnostics/
│   └── testing/
│       ├── hccl-test/
│       ├── ascend-dmi/
│       └── docker-env/
├── ascend-dev/                    # Development
│   ├── frameworks/                # AI Frameworks
│   │   ├── inference/             # Inference engines
│   │   │   ├── vllm/
│   │   │   │   ├── vllm-basic/
│   │   │   │   ├── vllm-parallel/
│   │   │   │   └── vllm-optimization/
│   │   │   ├── mindie/
│   │   │   └── sglang/
│   │   └── training/              # Training frameworks
│   │       ├── mindspeed/
│   │       ├── mindspore/
│   │       └── verl/
│   └── operators/                 # Custom operators
│       ├── catlass/
│       ├── ascendc/
│       ├── triton/
│       └── tilelang/
├── ascend-use-cases/              # Examples
│   ├── operator-cases/
│   ├── inference-cases/
│   └── training-cases/
└── ascend-utils/                  # Utilities
```

## Contributing

See [For Human Contributors](#for-human-contributors) section above for contribution guidelines.

## License

TODO: Add license
