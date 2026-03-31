//! Prompt construction for each role.
//!
//! Each module exports a builder function that takes the relevant payload
//! and returns a fully assembled prompt string. These are used differently
//! depending on the dispatch mode:
//!
//! - **CLI dispatch** (claude, codex): the prompt is the entire input to
//!   `claude -p` or `codex exec`. It includes role framing, task details,
//!   and output format instructions — everything in one string.
//!
//! - **Local dispatch** (orome-local): the prompt is split into system
//!   message + user message in a ChatRequest. The harness drives the
//!   tool loop, so tool definitions are injected separately.
//!
//! - **Interactive planner**: the system prompt is injected into the
//!   claude session on launch. The user's messages come from the REPL.

pub mod planner;
pub mod reviewer;
pub mod worker;
