/// @file Helper module for `Tensor` processing.
///
/// These functions and classes are only used internally,
/// meaning an end-user shouldn't need to access anything here.
///
/// @module utils/tensor



// const DataTypeMap = {
//   float32: Float32Array,
//   // @ts-ignore ts(2552) Limited availability of Float16Array across browsers:
//   // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float16Array
//   float16: typeof Float16Array !== "undefined" ? Float16Array: Uint16Array,
//   float64: Float64Array,
//   string: Array, // string[]
//   int8: Int8Array,
//   uint8: Uint8Array,
//   int16: Int16Array,
//   uint16: Uint16Array,
//   int32: Int32Array,
//   uint32: Uint32Array,
//   int64: BigInt64Array,
//   uint64: BigUint64Array,
//   bool: Uint8Array,
//   uint4: Uint8Array,
//   int4: Int8Array,
// };

/**
 * @typedef {keyof typeof DataTypeMap} DataType
 * @typedef {import('./maths.js').AnyTypedArray | any[]} DataArray
 */
