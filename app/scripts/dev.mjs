/**
 * Run the debugger's client and API server together, and take both down on Ctrl-C.
 *
 * The obvious shell version -- `trap 'kill 0' INT; npm run server & npm run
 * client:dev & wait` -- leaves zombies. Two reasons, both worth knowing before
 * anyone "simplifies" this back:
 *
 *   1. `npm run X` is a wrapper process. It does not forward signals to the
 *      binary it spawned, so signalling the wrapper orphans the real vite or
 *      node process, which keeps its port bound.
 *   2. `node --watch` runs the server in a grandchild process. Killing the node
 *      that owns the watcher does not touch the child actually holding :8182.
 *
 * So each child is spawned detached -- making it a process-group leader -- and
 * torn down by signalling the whole group (`kill(-pid)`), which reaches wrappers
 * and grandchildren alike. Vite and node are invoked directly rather than
 * through `npm run` to drop one wrapper layer. If a group ignores SIGTERM it is
 * SIGKILLed after a grace period, so Ctrl-C always returns a clean shell.
 */

import { spawn } from 'node:child_process';

const GRACE_MS = 4000;

/** One child process group: the command, and the handle we signal it through. */
const children = [];

function start(name, command, args) {
  const child = spawn(command, args, {
    // Its own process group, so one kill() reaches the wrapper and everything
    // it spawned rather than just the process we happen to hold a pid for.
    detached: true,
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  child.on('exit', (code, signal) => {
    // One half dying makes the other half useless (the client proxies to the
    // server), so a crash takes the whole dev environment down rather than
    // leaving a half-working setup that looks fine until an API call fails.
    if (!shuttingDown) {
      console.error(
        `\n[dev] ${name} exited (${signal ?? `code ${code}`}); stopping the rest.`,
      );
      shutdown(signal === 'SIGINT' ? 0 : 1);
    }
  });
  children.push({ name, child });
  return child;
}

let shuttingDown = false;

function signalGroup(child, signal) {
  try {
    // Negative pid targets the process group. The child may already be gone
    // between the exit event and this call, which is not an error.
    process.kill(-child.pid, signal);
  } catch {
    /* already dead */
  }
}

function shutdown(exitCode) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const { child } of children) signalGroup(child, 'SIGTERM');

  const deadline = setTimeout(() => {
    for (const { child } of children) signalGroup(child, 'SIGKILL');
    process.exit(exitCode);
  }, GRACE_MS);

  // Leave as soon as everything is actually gone, rather than always waiting
  // out the grace period.
  const poll = setInterval(() => {
    if (children.every(({ child }) => child.exitCode !== null || child.killed)) {
      clearTimeout(deadline);
      clearInterval(poll);
      process.exit(exitCode);
    }
  }, 100);
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => shutdown(0));
}

const serverArgs = process.argv.slice(2);
start('server', process.execPath, [
  '--watch-path=./server',
  '--watch-preserve-output',
  'server/main.ts',
  ...serverArgs,
]);
start('client', 'node_modules/.bin/vite', []);
