/** Whether a keyboard event originated in a field the user is typing into. */
export function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  return (
    tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable === true
  );
}
