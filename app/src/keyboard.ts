/** Whether a keyboard event originated in a field the user is typing into. */
export function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  return (
    tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable === true
  );
}

/** The next stop when a key cycles an opacity through 0/50/100%. */
export function nextOpacityStep(prev: number): number {
  const steps = [0, 50, 100];
  return steps[(steps.indexOf(prev) + 1) % steps.length] ?? 0;
}
