/**
 * Subagent Tools
 *
 * Two tools for spawning subagents with isolated context:
 * - subagent: single task (inherits the session model; task required)
 * - subagents: parallel tasks (tasks array, required; all inherit the session model)
 *
 * Split into separate tools so models see unambiguous schemas.
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
/**
 * Resolve the pi binary invocation, bypassing shell wrappers (e.g., sandbox).
 * Uses the same Node.js runtime and script that launched the current process.
 */
function getPiInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	if (currentScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}

	const execName = path.basename(process.execPath).toLowerCase();
	const isGenericRuntime = /^(node|bun)(\.exe)?$/.test(execName);
	if (!isGenericRuntime) {
		return { command: process.execPath, args };
	}

	return { command: "pi", args };
}
import { type ExtensionAPI, type ExtensionContext, getMarkdownTheme, getLanguageFromPath, highlightCode } from "@earendil-works/pi-coding-agent";
import { Type, type Message } from "@earendil-works/pi-ai";
import { Container, Markdown, Spacer, Text } from "@earendil-works/pi-tui";

const MAX_PARALLEL = 32;

/** Custom entry type for persisting subagent usage */
const SUBAGENT_USAGE_ENTRY_TYPE = "subagent_usage";
const MAX_CONCURRENCY = 4;

interface UsageStats {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cost: number;
	turns: number;
}

/** Data stored in custom session entries for subagent usage tracking */
interface SubagentUsageEntry {
	usage: UsageStats;
	model: string;
	timestamp: number;
}

interface SubagentResult {
	model: string;
	task: string;
	context?: string;
	exitCode: number;
	output: string;
	messages: Message[];
	/** Partial message during streaming (contains in-progress tool calls) */
	partialMessage?: Message;
	usage: UsageStats;
	stopReason?: string;
	errorMessage?: string;
}

interface SubagentDetails {
	mode: "single" | "parallel";
	results: SubagentResult[];
}

/** Check if a subagent result is an error (used consistently throughout) */
function isResultError(r: SubagentResult): boolean {
	return r.exitCode !== 0 || r.stopReason === "error" || r.stopReason === "aborted";
}

async function mapWithConcurrency<T, R>(
	items: T[],
	concurrency: number,
	fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
	const results: R[] = new Array(items.length);
	let nextIndex = 0;
	const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
		while (nextIndex < items.length) {
			const index = nextIndex++;
			results[index] = await fn(items[index], index);
		}
	});
	await Promise.all(workers);
	return results;
}

interface ToolCallWithResult {
	type: "toolCall";
	id: string;
	name: string;
	args: Record<string, unknown>;
	result?: {
		content: Array<{ type: string; text?: string }>;
		isError: boolean;
	};
}

type DisplayItem = 
	| { type: "text"; text: string }
	| ToolCallWithResult;

function getDisplayItems(messages: Message[]): DisplayItem[] {
	const items: DisplayItem[] = [];
	// First pass: collect tool calls
	const toolCalls = new Map<string, ToolCallWithResult>();
	
	for (const msg of messages) {
		if (msg.role === "assistant") {
			for (const part of msg.content) {
				if (part.type === "text") {
					items.push({ type: "text", text: part.text });
				} else if (part.type === "toolCall") {
					const tc: ToolCallWithResult = {
						type: "toolCall",
						id: part.id,
						name: part.name,
						args: part.arguments
					};
					toolCalls.set(part.id, tc);
					items.push(tc);
				}
			}
		} else if (msg.role === "toolResult") {
			// Match result to its call
			const tc = toolCalls.get(msg.toolCallId);
			if (tc) {
				tc.result = {
					content: msg.content as Array<{ type: string; text?: string }>,
					isError: msg.isError
				};
			}
		}
	}
	return items;
}

function getFinalOutput(messages: Message[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (msg.role === "assistant") {
			for (const part of msg.content) {
				if (part.type === "text") return part.text;
			}
		}
	}
	return "";
}

/** Get messages for display, including partial streaming message if present */
function getDisplayMessages(r: SubagentResult): Message[] {
	return r.partialMessage ? [r.partialMessage, ...r.messages] : r.messages;
}

function shortenPath(p: string): string {
	const home = os.homedir();
	return p.startsWith(home) ? `~${p.slice(home.length)}` : p;
}

function formatToolCall(name: string, args: Record<string, unknown>, themeFg: (color: string, text: string) => string): string {
	switch (name.toLowerCase()) {
		case "bash": {
			const cmd = (args.command as string) || "...";
			const preview = cmd.length > 60 ? `${cmd.slice(0, 60)}...` : cmd;
			return themeFg("muted", "$ ") + themeFg("toolOutput", preview);
		}
		case "read": {
			const filePath = shortenPath((args.path || args.file_path || "...") as string);
			return themeFg("muted", "read ") + themeFg("accent", filePath);
		}
		case "write": {
			const filePath = shortenPath((args.path || args.file_path || "...") as string);
			return themeFg("muted", "write ") + themeFg("accent", filePath);
		}
		case "edit": {
			const filePath = shortenPath((args.path || args.file_path || "...") as string);
			return themeFg("muted", "edit ") + themeFg("accent", filePath);
		}
		default: {
			const argsStr = JSON.stringify(args);
			const preview = argsStr.length > 50 ? `${argsStr.slice(0, 50)}...` : argsStr;
			return themeFg("accent", name) + themeFg("dim", ` ${preview}`);
		}
	}
}

/** Render a tool call with full details (for expanded view) */
function renderToolCallExpanded(
	name: string,
	args: Record<string, unknown>,
	result: { content: Array<{ type: string; text?: string }>; isError: boolean } | undefined,
	theme: { fg: (color: string, text: string) => string; bold: (text: string) => string },
	fullyExpanded: boolean = false
): Container {
	const container = new Container();
	const lowerName = name.toLowerCase();

	// Helper to get text from result
	const getResultText = (): string => {
		if (!result) return "";
		return result.content
			.filter(c => c.type === "text" && c.text)
			.map(c => c.text!)
			.join("\n");
	};

	switch (lowerName) {
		case "bash": {
			const cmd = (args.command as string) || "...";
			const timeout = args.timeout as number | undefined;
			let header = theme.fg("muted", "$ ") + theme.fg("toolOutput", cmd);
			if (timeout) header += theme.fg("dim", ` (timeout ${timeout}s)`);
			container.addChild(new Text(header, 0, 0));
			// Show bash output
			const output = getResultText();
			if (output) {
				const allLines = output.split("\n");
				const totalLines = allLines.length;
				// Filter to non-empty lines for display, but count from total
				const nonEmptyLines = allLines.filter(l => l.trim() !== "");
				const maxLines = fullyExpanded ? nonEmptyLines.length : 10;
				const displayLines = nonEmptyLines.slice(0, maxLines);
				const remaining = totalLines - maxLines;
				if (displayLines.length > 0) {
					container.addChild(new Text(theme.fg("dim", displayLines.join("\n")), 0, 0));
				}
				if (remaining > 0 && !fullyExpanded) {
					container.addChild(new Text(theme.fg("muted", `... (${remaining} more lines, ${totalLines} total)`), 0, 0));
				}
			}
			break;
		}
		case "read": {
			const rawPath = (args.path || args.file_path || "...") as string;
			const filePath = shortenPath(rawPath);
			const offset = args.offset as number | undefined;
			const limit = args.limit as number | undefined;
			let pathDisplay = theme.fg("accent", filePath);
			if (offset !== undefined || limit !== undefined) {
				const startLine = offset ?? 1;
				const endLine = limit !== undefined ? startLine + limit - 1 : "";
				pathDisplay += theme.fg("warning", `:${startLine}${endLine ? `-${endLine}` : ""}`);
			}
			container.addChild(new Text(theme.fg("muted", "read ") + pathDisplay, 0, 0));
			// Show file content from result
			const content = getResultText();
			if (content) {
				const lang = getLanguageFromPath(rawPath);
				const allLines = content.split("\n");
				const totalLines = allLines.length;
				const maxLines = fullyExpanded ? allLines.length : 15;
				const displayLines = allLines.slice(0, maxLines);
				const remaining = totalLines - maxLines;
				const displayText = displayLines.join("\n");
				if (displayText.trim()) {
					try {
						if (lang) {
							const highlighted = highlightCode(displayText, lang);
							container.addChild(new Text(highlighted.join("\n"), 0, 0));
						} else {
							container.addChild(new Text(theme.fg("toolOutput", displayText), 0, 0));
						}
					} catch {
						container.addChild(new Text(theme.fg("toolOutput", displayText), 0, 0));
					}
				}
				if (remaining > 0 && !fullyExpanded) {
					container.addChild(new Text(theme.fg("muted", `... (${remaining} more lines, ${totalLines} total)`), 0, 0));
				}
			}
			break;
		}
		case "write": {
			const rawPath = (args.path || args.file_path || "...") as string;
			const filePath = shortenPath(rawPath);
			const content = (args.content as string) || "";
			container.addChild(new Text(theme.fg("muted", "write ") + theme.fg("accent", filePath), 0, 0));
			if (content) {
				const allLines = content.split("\n");
				const totalLines = allLines.length;
				const maxLines = fullyExpanded ? allLines.length : 15;
				const displayLines = allLines.slice(0, maxLines);
				const remaining = totalLines - maxLines;
				const displayText = displayLines.join("\n");
				
				// Try syntax highlighting, fall back to plain
				if (displayText.trim()) {
					const lang = getLanguageFromPath(rawPath);
					try {
						if (lang) {
							const highlighted = highlightCode(displayText, lang);
							container.addChild(new Text(highlighted.join("\n"), 0, 0));
						} else {
							container.addChild(new Text(theme.fg("toolOutput", displayText), 0, 0));
						}
					} catch {
						container.addChild(new Text(theme.fg("toolOutput", displayText), 0, 0));
					}
				}
				if (remaining > 0 && !fullyExpanded) {
					container.addChild(new Text(theme.fg("muted", `... (${remaining} more lines, ${totalLines} total)`), 0, 0));
				}
			}
			break;
		}
		case "edit": {
			const rawPath = (args.path || args.file_path || "...") as string;
			const filePath = shortenPath(rawPath);
			const oldText = (args.oldText as string) || "";
			const newText = (args.newText as string) || "";
			container.addChild(new Text(theme.fg("muted", "edit ") + theme.fg("accent", filePath), 0, 0));
			
			// Show old/new text as simple diff-like display
			if (oldText || newText) {
				const allOldLines = oldText.split("\n");
				const allNewLines = newText.split("\n");
				const oldTotalLines = allOldLines.length;
				const newTotalLines = allNewLines.length;
				const oldMaxLines = fullyExpanded ? oldTotalLines : 10;
				const newMaxLines = fullyExpanded ? newTotalLines : 10;
				const oldDisplayLines = allOldLines.slice(0, oldMaxLines);
				const newDisplayLines = allNewLines.slice(0, newMaxLines);
				const oldRemaining = oldTotalLines - oldDisplayLines.length;
				const newRemaining = newTotalLines - newDisplayLines.length;
				
				if (oldText) {
					const oldFormatted = oldDisplayLines.map(l => theme.fg("error", "- " + l)).join("\n");
					if (oldFormatted.trim()) {
						container.addChild(new Text(oldFormatted, 0, 0));
					}
					if (oldRemaining > 0 && !fullyExpanded) {
						container.addChild(new Text(theme.fg("muted", `  ... (${oldRemaining} more lines, ${oldTotalLines} total)`), 0, 0));
					}
				}
				if (newText) {
					const newFormatted = newDisplayLines.map(l => theme.fg("success", "+ " + l)).join("\n");
					if (newFormatted.trim()) {
						container.addChild(new Text(newFormatted, 0, 0));
					}
					if (newRemaining > 0 && !fullyExpanded) {
						container.addChild(new Text(theme.fg("muted", `  ... (${newRemaining} more lines, ${newTotalLines} total)`), 0, 0));
					}
				}
			}
			break;
		}
		default: {
			// For unknown tools, show name and formatted JSON args
			container.addChild(new Text(theme.fg("accent", theme.bold(name)), 0, 0));
			const argsStr = JSON.stringify(args, null, 2);
			const allLines = argsStr.split("\n");
			const totalLines = allLines.length;
			const maxLines = fullyExpanded ? totalLines : 20;
			const displayLines = allLines.slice(0, maxLines);
			const remaining = totalLines - displayLines.length;
			const displayText = displayLines.join("\n");
			if (displayText.trim()) {
				container.addChild(new Text(theme.fg("dim", displayText), 0, 0));
			}
			if (remaining > 0 && !fullyExpanded) {
				container.addChild(new Text(theme.fg("muted", `... (${remaining} more lines, ${totalLines} total)`), 0, 0));
			}
			break;
		}
	}

	return container;
}

function formatTokens(n: number): string {
	if (n < 1000) return n.toString();
	if (n < 10000) return `${(n / 1000).toFixed(1)}k`;
	return `${Math.round(n / 1000)}k`;
}

function formatUsage(u: UsageStats, model: string): string {
	const parts: string[] = [];
	if (u.turns) parts.push(`${u.turns} turn${u.turns > 1 ? "s" : ""}`);
	if (u.input) parts.push(`↑${formatTokens(u.input)}`);
	if (u.output) parts.push(`↓${formatTokens(u.output)}`);
	if (u.cacheRead) parts.push(`R${formatTokens(u.cacheRead)}`);
	if (u.cacheWrite) parts.push(`W${formatTokens(u.cacheWrite)}`);
	if (u.cost) parts.push(`$${u.cost.toFixed(4)}`);
	parts.push(model);
	return parts.join(" ");
}

async function runSubagent(
	cwd: string,
	model: { provider: string; id: string; display: string },
	task: string,
	context: string | undefined,
	tools: string[] | undefined,
	signal: AbortSignal | undefined,
	onUpdate: ((result: SubagentResult) => void) | undefined,
): Promise<SubagentResult> {
	// Pass provider and model as separate CLI flags. A provider/id string can be
	// ambiguous when multiple providers serve the same model id (or model ids
	// themselves contain slashes), so the provider must be preserved explicitly.
	const args = ["--mode", "json", "-p", "--no-session", "--provider", model.provider, "--model", model.id];

	// Let the subagent know it's a subagent to discourage recursive spawning
	args.push("--append-system-prompt", "You are a subagent. Complete your task directly.");

	if (tools && tools.length > 0) {
		// Tool names are case-sensitive (lowercase)
		args.push("--tools", tools.map(t => t.toLowerCase()).join(","));
	}

	// Build the prompt
	let prompt = "";
	if (context) {
		prompt += `<context>\n${context}\n</context>\n\n`;
	}
	prompt += `Task: ${task}`;
	args.push(prompt);

	const result: SubagentResult = {
		model: model.display,
		task,
		context,
		exitCode: -1, // -1 = still running
		output: "",
		messages: [],
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
	};

	const emitUpdate = () => onUpdate?.(result);

	let wasAborted = false;

	const exitCode = await new Promise<number>((resolve) => {
		const invocation = getPiInvocation(args);
		const proc = spawn(invocation.command, invocation.args, {
			cwd,
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
			env: { ...process.env, FORCE_COLOR: "0", NO_COLOR: "1", PI_SUBAGENT: "1" },
		});
		let buffer = "";
		let stderr = "";

		const processLine = (line: string) => {
			if (!line.trim()) return;
			let event: any;
			try {
				event = JSON.parse(line);
			} catch {
				return;
			}

			if (event.type === "message_end" && event.message) {
				const msg = event.message as Message;
				result.messages.push(msg);
				// Clear partial - it's now in messages
				result.partialMessage = undefined;
				if (msg.role === "assistant") {
					result.usage.turns++;
					// Update output to latest text
					result.output = getFinalOutput(result.messages);
					const usage = msg.usage;
					if (usage) {
						result.usage.input += usage.input || 0;
						result.usage.output += usage.output || 0;
						result.usage.cacheRead += usage.cacheRead || 0;
						result.usage.cacheWrite += usage.cacheWrite || 0;
						result.usage.cost += usage.cost?.total || 0;
					}
					if (msg.stopReason) result.stopReason = msg.stopReason;
					if (msg.errorMessage) result.errorMessage = msg.errorMessage;
				}
				emitUpdate();
			}

			// Capture tool calls as they complete (for intermediate traces)
			if (event.type === "message_update" && event.assistantMessageEvent?.type === "toolcall_end") {
				// Update the partial message in-place for intermediate display
				result.partialMessage = event.message as Message;
				emitUpdate();
			}

			// Capture nested subagent costs from tool results
			if (event.type === "tool_execution_end" && event.toolName === "subagent") {
				const details = event.result?.details as SubagentDetails | undefined;
				if (details?.results) {
					for (const r of details.results) {
						if (r.usage) {
							result.usage.input += r.usage.input || 0;
							result.usage.output += r.usage.output || 0;
							result.usage.cacheRead += r.usage.cacheRead || 0;
							result.usage.cacheWrite += r.usage.cacheWrite || 0;
							result.usage.cost += r.usage.cost || 0;
							result.usage.turns += r.usage.turns || 0;
						}
					}
				}
			}


		};

		proc.stdout.on("data", (data) => {
			buffer += data.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() || "";
			for (const line of lines) processLine(line);
		});

		proc.stderr.on("data", (data) => {
			stderr += data.toString();
		});

		let hasExited = false;
		let killTimeout: ReturnType<typeof setTimeout> | undefined;
		
		const abortHandler = () => {
			wasAborted = true;
			proc.kill("SIGTERM");
			killTimeout = setTimeout(() => {
				if (!hasExited) proc.kill("SIGKILL");
			}, 3000);
		};

		proc.on("close", (code, sig) => {
			hasExited = true;
			if (killTimeout) clearTimeout(killTimeout);
			if (signal) signal.removeEventListener("abort", abortHandler);
			
			if (buffer.trim()) processLine(buffer);
			
			// Handle signal kills (code is null when killed by signal)
			if (sig) {
				if (!result.errorMessage) result.errorMessage = `Killed by ${sig}`;
				resolve(1);
			} else if (code !== 0 && !result.errorMessage) {
				result.errorMessage = stderr.trim() || `Exit code ${code}`;
				resolve(code ?? 1);
			} else {
				resolve(code ?? 0);
			}
		});

		proc.on("error", (err) => {
			result.errorMessage = err.message;
			resolve(1);
		});

		if (signal) {
			if (signal.aborted) abortHandler();
			else signal.addEventListener("abort", abortHandler, { once: true });
		}
	});

	result.exitCode = exitCode;
	if (wasAborted) {
		result.stopReason = "aborted";
		result.errorMessage = "Aborted by user";
	}

	return result;
}

export default function (pi: ExtensionAPI) {
	// Subagents cannot spawn further subagents - skip tool registration
	if (process.env.PI_SUBAGENT === "1") return;
	// Track cumulative subagent usage across the session
	let cumulativeUsage: UsageStats = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 };

	// Helper to record usage and update status
	const recordUsage = (usage: UsageStats, model: string, ctx: ExtensionContext) => {
		cumulativeUsage.input += usage.input;
		cumulativeUsage.output += usage.output;
		cumulativeUsage.cacheRead += usage.cacheRead;
		cumulativeUsage.cacheWrite += usage.cacheWrite;
		cumulativeUsage.cost += usage.cost;
		cumulativeUsage.turns += usage.turns;

		// Persist to session
		pi.appendEntry<SubagentUsageEntry>(SUBAGENT_USAGE_ENTRY_TYPE, {
			usage,
			model,
			timestamp: Date.now(),
		});

		// Update footer status
		if (cumulativeUsage.cost > 0) {
			ctx.ui.setStatus("subagent", `subagents: $${cumulativeUsage.cost.toFixed(3)}`);
		}
	};

	// Restore cumulative usage from session entries on load
	const restoreUsageFromSession = (ctx: ExtensionContext) => {
		cumulativeUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 };
		for (const entry of ctx.sessionManager.getEntries()) {
			if (entry.type === "custom" && entry.customType === SUBAGENT_USAGE_ENTRY_TYPE) {
				const data = entry.data as SubagentUsageEntry;
				if (data?.usage) {
					cumulativeUsage.input += data.usage.input;
					cumulativeUsage.output += data.usage.output;
					cumulativeUsage.cacheRead += data.usage.cacheRead;
					cumulativeUsage.cacheWrite += data.usage.cacheWrite;
					cumulativeUsage.cost += data.usage.cost;
					cumulativeUsage.turns += data.usage.turns;
				}
			}
		}
		if (cumulativeUsage.cost > 0) {
			ctx.ui.setStatus("subagent", `subagents: $${cumulativeUsage.cost.toFixed(3)}`);
		}
	};

	// Restore on session start
	pi.on("session_start", (_, ctx) => {
		restoreUsageFromSession(ctx);
	});

	// Resolve the current session model. Subagents always inherit both provider and
	// model id and do not accept explicit model overrides.
	const resolveSessionModel = (ctx: ExtensionContext): { provider: string; id: string; display: string } => {
		const m = ctx.model;
		if (!m) {
			throw new Error("The session has no active model; subagents inherit the session model.");
		}
		return { provider: m.provider, id: m.id, display: `${m.provider}/${m.id}` };
	};

	// Shared renderResult for both tools (already dispatches on details.mode)
	const sharedRenderResult = (result: any, { expanded }: { expanded: boolean }, theme: any) => {
		const details = result.details as SubagentDetails | undefined;
		if (!details || details.results.length === 0) {
			const text = result.content[0];
			return new Text(text?.type === "text" ? text.text : "(no output)", 0, 0);
		}

		const mdTheme = getMarkdownTheme();

		const aggregateUsage = (results: SubagentResult[]): UsageStats => {
			const total = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 };
			for (const r of results) {
				total.input += r.usage.input;
				total.output += r.usage.output;
				total.cacheRead += r.usage.cacheRead;
				total.cacheWrite += r.usage.cacheWrite;
				total.cost += r.usage.cost;
				total.turns += r.usage.turns;
			}
			return total;
		};

		const renderSingleResult = (r: SubagentResult, showHeader: boolean, showExpanded: boolean) => {
			const isError = r.exitCode !== -1 && isResultError(r);
			const icon = r.exitCode === -1
				? theme.fg("warning", "⏳")
				: isError ? theme.fg("error", "✗") : theme.fg("success", "✓");
			const displayItems = getDisplayItems(getDisplayMessages(r));
			const finalOutput = getFinalOutput(r.messages);
			const toolCalls = displayItems.filter((i: DisplayItem) => i.type === "toolCall");

			const container = new Container();

			if (showHeader) {
				let header = `${icon} ${theme.fg("accent", r.model)}`;
				if (isError && r.stopReason) header += ` ${theme.fg("error", `[${r.stopReason}]`)}`;
				container.addChild(new Text(header, 0, 0));
			}

			if (r.errorMessage) {
				container.addChild(new Text(theme.fg("error", r.errorMessage), 0, 0));
			}

			// Task
			container.addChild(new Text(theme.fg("muted", "Task: ") + theme.fg("dim", r.task), 0, 0));

			// Tool calls
			for (const item of toolCalls) {
				if (item.type === "toolCall") {
					if (showExpanded) {
						container.addChild(new Spacer(1));
						container.addChild(renderToolCallExpanded(item.name, item.args, item.result, theme, true));
					} else {
						container.addChild(new Text(
							theme.fg("muted", "→ ") + formatToolCall(item.name, item.args, theme.fg.bind(theme)),
							0, 0
						));
					}
				}
			}

			// Output
			if (finalOutput) {
				container.addChild(new Spacer(1));
				container.addChild(new Markdown(finalOutput.trim(), 0, 0, mdTheme));
			} else if (r.exitCode === -1) {
				container.addChild(new Text(theme.fg("muted", "(running...)"), 0, 0));
			}

			// Usage
			if (r.exitCode !== -1) {
				container.addChild(new Text(theme.fg("dim", formatUsage(r.usage, r.model)), 0, 0));
			}

			return container;
		};

		// Single mode
		if (details.mode === "single") {
			const r = details.results[0];
			const isRunning = r.exitCode === -1;
			const isError = !isRunning && isResultError(r);
			const icon = isRunning
				? theme.fg("warning", "⏳")
				: isError ? theme.fg("error", "✗") : theme.fg("success", "✓");

			if (expanded) {
				const container = new Container();
				const displayItems = getDisplayItems(getDisplayMessages(r));
				const toolCalls = displayItems.filter((i: DisplayItem) => i.type === "toolCall");

				if (isError) {
					let statusLine = icon;
					if (r.stopReason) statusLine += ` ${theme.fg("error", `[${r.stopReason}]`)}`;
					if (r.errorMessage) statusLine += ` ${theme.fg("error", r.errorMessage)}`;
					container.addChild(new Text(statusLine, 0, 0));
				}

				if (r.context) {
					container.addChild(new Spacer(1));
					container.addChild(new Text(theme.fg("muted", "─── Context ───"), 0, 0));
					container.addChild(new Text(theme.fg("dim", r.context), 0, 0));
				}

				if (toolCalls.length > 0) {
					container.addChild(new Spacer(1));
					container.addChild(new Text(theme.fg("muted", "─── Tool Calls ───"), 0, 0));
					for (const item of toolCalls) {
						if (item.type === "toolCall") {
							container.addChild(new Spacer(1));
							container.addChild(renderToolCallExpanded(item.name, item.args, item.result, theme, true));
						}
					}
				}

				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Output ───"), 0, 0));
				const finalOut = getFinalOutput(r.messages);
				if (finalOut) {
					container.addChild(new Markdown(finalOut.trim(), 0, 0, mdTheme));
				} else {
					container.addChild(new Text(theme.fg("muted", "(no output)"), 0, 0));
				}

				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("dim", formatUsage(r.usage, r.model)), 0, 0));

				return container;
			}

			// Collapsed single
			let text = `${icon} ${theme.fg("accent", r.model)}`;
			if (isError && r.errorMessage) {
				text += ` ${theme.fg("error", r.errorMessage)}`;
			} else if (r.output) {
				const allLines = r.output.split("\n");
				const totalLines = allLines.length;
				const nonEmptyLines = allLines.filter((l: string) => l.trim() !== "");
				const previewLines = nonEmptyLines.slice(0, 5);
				const remaining = totalLines - 5;
				if (previewLines.length > 0) {
					text += "\n" + theme.fg("toolOutput", previewLines.join("\n"));
				}
				if (remaining > 0) {
					text += "\n" + theme.fg("muted", `... (${remaining} more lines, Ctrl+O to expand)`);
				}
			} else {
				text += " " + theme.fg("muted", "(no output)");
			}
			text += "\n" + theme.fg("dim", formatUsage(r.usage, r.model));
			return new Text(text, 0, 0);
		}

		// Parallel mode
		const running = details.results.filter((r) => r.exitCode === -1).length;
		const done = details.results.filter((r) => r.exitCode !== -1);
		const successCount = done.filter((r) => !isResultError(r)).length;
		const failCount = done.filter((r) => isResultError(r)).length;
		const isRunning = running > 0;
		const icon = isRunning
			? theme.fg("warning", "⏳")
			: failCount > 0
				? theme.fg("warning", "◐")
				: theme.fg("success", "✓");
		const status = isRunning
			? `${successCount + failCount}/${details.results.length} done, ${running} running`
			: `${successCount}/${details.results.length} succeeded`;

		if (expanded) {
			const container = new Container();
			container.addChild(new Text(
				`${icon} ${theme.fg("toolTitle", theme.bold("subagents "))}${theme.fg("accent", `parallel ${status}`)}`,
				0, 0
			));

			for (const r of details.results) {
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "────────────────────"), 0, 0));
				container.addChild(renderSingleResult(r, true, true));
			}

			if (!isRunning) {
				const totalUsage = aggregateUsage(details.results);
				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("dim", `Total: ${formatUsage(totalUsage, "")}`), 0, 0));
			}

			return container;
		}

		// Collapsed parallel
		let text = `${icon} ${theme.fg("toolTitle", theme.bold("parallel "))}${theme.fg("accent", status)}`;
		for (const r of details.results) {
			const rIcon = r.exitCode === -1
				? theme.fg("warning", "⏳")
				: isResultError(r) ? theme.fg("error", "✗") : theme.fg("success", "✓");
			let preview: string;
			if (r.output) {
				preview = (r.output.length > 60 ? r.output.slice(0, 60) + "..." : r.output).split("\n")[0];
			} else if (r.exitCode === -1) {
				const toolCalls = getDisplayItems(getDisplayMessages(r)).filter((i: DisplayItem) => i.type === "toolCall");
				if (toolCalls.length > 0) {
					const last = toolCalls[toolCalls.length - 1];
					const name = last.name.toLowerCase();
					if (name === "bash") {
						const cmd = (last.args.command as string) ?? "";
						preview = `$ ${cmd.slice(0, 40)}${cmd.length > 40 ? "..." : ""}`;
					} else if (name === "read" || name === "write" || name === "edit") {
						const p = shortenPath((last.args.path || last.args.file_path || "") as string);
						preview = `${name} ${p}`;
					} else {
						preview = `${name}...`;
					}
				} else {
					preview = "(starting...)";
				}
			} else {
				preview = "(no output)";
			}
			text += `\n${rIcon} ${theme.fg("accent", r.model)} ${theme.fg("dim", preview)}`;
		}
		if (!isRunning) {
			const totalUsage = aggregateUsage(details.results);
			text += `\n${theme.fg("dim", `Total: ${formatUsage(totalUsage, "")}`)}`;
		}
		if (!expanded) {
			text += `\n${theme.fg("muted", "(Ctrl+O to expand)")}`;
		}
		return new Text(text, 0, 0);
	};

	// ── Single subagent tool ──
	pi.registerTool({
		name: "subagent",
		label: "Subagent",
		description:
			"Spawn a subagent with isolated context. The subagent always inherits the current session model. Params: task, context (optional), tools (optional array).",
		parameters: Type.Object({
			task: Type.String({ description: "The task instruction for the subagent" }),
			context: Type.Optional(Type.String({ description: "Optional XML-structured context to pass" })),
			tools: Type.Optional(Type.Array(Type.String(), { description: "Tool names to enable (default: all)" })),
		}),

		async execute(_id, params, signal, onUpdate, ctx) {
			const modelSpec = resolveSessionModel(ctx);

			const result = await runSubagent(
				ctx.cwd,
				modelSpec,
				params.task,
				params.context,
				params.tools,
				signal,
				onUpdate
					? (r) =>
							onUpdate({
								content: [{ type: "text", text: r.output || "(running...)" }],
								details: { mode: "single", results: [r] } as SubagentDetails,
							})
					: undefined,
			);

			if (result.usage.cost > 0 || result.usage.input > 0) {
				recordUsage(result.usage, result.model, ctx);
			}

			// Throw on subagent failure so the agent loop marks isError
			if (isResultError(result)) {
				throw new Error(result.errorMessage || result.output || "Subagent failed");
			}

			return {
				content: [{ type: "text", text: result.output || "(no output)" }],
				details: { mode: "single", results: [result] } as SubagentDetails,
			};
		},

		renderCall(args, theme) {
			const task = args.task || "...";

			let text = theme.fg("toolTitle", theme.bold("subagent "));
			text += theme.fg("accent", "session model");
			if (args.tools?.length) {
				text += theme.fg("muted", ` [${args.tools.join(", ")}]`);
			}
			text += "\n" + theme.fg("dim", task);
			if (args.context) {
				const lines = args.context.split("\n").length;
				text += "\n" + theme.fg("muted", `(+${lines} lines context)`);
			}

			return new Text(text, 0, 0);
		},

		renderResult: sharedRenderResult,
	});

	// ── Parallel subagents tool ──
	const TaskItem = Type.Object({
		task: Type.String({ description: "Task instruction" }),
		context: Type.Optional(Type.String({ description: "Optional XML context" })),
		tools: Type.Optional(Type.Array(Type.String(), { description: "Tool names to enable" })),
	});

	pi.registerTool({
		name: "subagents",
		label: "Subagents (parallel)",
		description:
			"Spawn multiple subagents in parallel. Each task runs concurrently with isolated context and inherits the current session model.",
		parameters: Type.Object({
			tasks: Type.Array(TaskItem, { description: `Array of tasks for parallel execution (max ${MAX_PARALLEL})`, minItems: 1 }),
		}),

		async execute(_id, params, signal, onUpdate, ctx) {
			if (params.tasks.length > MAX_PARALLEL) {
				throw new Error(`Too many tasks (${params.tasks.length}). Max is ${MAX_PARALLEL}.`);
			}

			const sessionModel = resolveSessionModel(ctx);

			const allResults: SubagentResult[] = params.tasks.map((t) => ({
				model: sessionModel.display,
				task: t.task,
				context: t.context,
				exitCode: -1,
				output: "",
				messages: [],
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
			}));

			const emitUpdate = () => {
				if (onUpdate) {
					const done = allResults.filter((r) => r.exitCode !== -1).length;
					const running = allResults.length - done;
					onUpdate({
						content: [{ type: "text", text: `${done}/${allResults.length} done, ${running} running...` }],
						details: { mode: "parallel", results: allResults } as SubagentDetails,
					});
				}
			};

			emitUpdate();

			await mapWithConcurrency(params.tasks, MAX_CONCURRENCY, async (t: { task: string; context?: string; tools?: string[] }, index) => {
				const result = await runSubagent(
					ctx.cwd,
					sessionModel,
					t.task,
					t.context,
					t.tools,
					signal,
					(r) => {
						allResults[index] = r;
						emitUpdate();
					},
				);
				allResults[index] = result;
				emitUpdate();
				return result;
			});

			const successCount = allResults.filter((r) => !isResultError(r)).length;
			const fullOutputs = allResults.map((r, i) => {
				const status = isResultError(r) ? "✗" : "✓";
				const header = `[${i + 1}/${allResults.length}] ${status} ${r.model}`;
				const body = r.output || r.errorMessage || "(no output)";
				return `${header}\n${body}`;
			});

			for (const r of allResults) {
				if (r.usage.cost > 0 || r.usage.input > 0) {
					recordUsage(r.usage, r.model, ctx);
				}
			}

			// Don't throw on partial failure — report results, let model decide
			return {
				content: [{ type: "text", text: `${successCount}/${allResults.length} succeeded\n\n${fullOutputs.join("\n\n---\n\n")}` }],
				details: { mode: "parallel", results: allResults } as SubagentDetails,
			};
		},

		renderCall(args, theme) {
			let text = theme.fg("toolTitle", theme.bold("subagents "));
			const tasks = args.tasks || [];
			text += theme.fg("accent", `parallel (${tasks.length} tasks)`);
			for (const t of tasks.slice(0, 3)) {
				const preview = t.task.length > 40 ? t.task.slice(0, 40) + "..." : t.task;
				text += `\n  ${theme.fg("dim", preview)}`;
			}
			if (tasks.length > 3) {
				text += `\n  ${theme.fg("muted", `... +${tasks.length - 3} more`)}`;
			}
			return new Text(text, 0, 0);
		},

		renderResult: sharedRenderResult,
	});

	// Command: /subagent <task>
	pi.registerCommand("subagent", {
		description: "Delegate to a subagent: /subagent <task>",
		handler: async (args, ctx) => {
			if (!args?.trim()) {
				ctx.ui.notify("Usage: /subagent <task>", "info");
				return;
			}

			pi.sendUserMessage(`Use a subagent: ${args.trim()}`);
		},
	});
}
