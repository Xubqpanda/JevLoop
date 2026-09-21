/**
 * 编译期穷尽性检查。
 *
 * 覆盖封闭联合的 `switch` 在最后写 `default: return assertNever(x)`，
 * 将来给联合加成员时编译器会在**所有**没处理它的地方报错，
 * 而不是让它静默走到 default。这是 JevLoop/AGENTS.md §5 的要求 ——
 * 在写这个文件之前，那条规范在仓库里零效力（没有任何地方定义过 `assertNever`）。
 *
 * 运行时它只抛错，不返回。参数类型是 `never`，正常代码传不进来。
 *
 * @module JevLoop/util
 */

/**
 * 断言一个值不可能存在。**只在覆盖封闭联合的 `switch` 的 `default` 分支使用。**
 *
 * 不要用它做运行时校验 —— 它存在的意义是让**编译器**在联合扩展时报错，
 * 不是处理非法输入。真实边界的校验见 AGENTS.md §6。
 *
 * @param value 被所有 `case` 排除后剩下的值，类型应为 `never`
 * @returns 永不返回，总是抛出
 */
export function assertNever(value: never): never {
  throw new Error(`未处理的联合成员：${JSON.stringify(value)}`)
}
