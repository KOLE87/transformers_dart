sealed class ProgressInfo {
  String get status => throw UnimplementedError('Implement status');
}

final class InitiateProgressInfo extends ProgressInfo {
  @override
  String get status => 'initiate';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  InitiateProgressInfo(this.name, this.file);
}

final class DownloadProgressInfo extends ProgressInfo {
  @override
  String get status => 'download';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  DownloadProgressInfo(this.name, this.file);
}

final class ProgressStatusInfo extends ProgressInfo {
  @override
  String get status => 'progress';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  /// A number between 0 and 100.
  num progress;

  /// The number of bytes loaded.
  num loaded;

  /// The total number of bytes to be loaded.
  num total;

  ProgressStatusInfo(this.name, this.file, this.progress, this.loaded, this.total);
}

final class DoneProgressInfo extends ProgressInfo {
  @override
  String get status => 'done';

  /// The model id or directory path.
  String name;

  /// The name of the file.
  String file;

  DoneProgressInfo(this.name, this.file);
}

final class ReadyProgressInfo extends ProgressInfo {
  @override
  String get status => 'ready';

  /// The loaded task.
  String task;

  /// The loaded model.
  String model;

  ReadyProgressInfo(this.task, this.model);
}

/**
 * @typedef {InitiateProgressInfo | DownloadProgressInfo | ProgressStatusInfo | DoneProgressInfo | ReadyProgressInfo} ProgressInfo
 */

/// A callback function that is called with progress information.
typedef ProgressCallback = void Function(ProgressInfo progressInfo);

/// Helper function to dispatch progress callbacks.
///
/// @param {ProgressCallback | null | undefined} progress_callback The progress callback function to dispatch.
/// @param {ProgressInfo} data The data to pass to the progress callback function.
/// @returns {void}
/// @private
void dispatchCallback(ProgressCallback? progress_callback, ProgressInfo data) {
  if (progress_callback != null) progress_callback(data);
}

///
/// @param {Object} o
/// @param {string[]} props
/// @returns {Object}
Map<String, dynamic> pick(Map<String, dynamic> o, List<String> props) {
  final Map<String, dynamic> result = {};
  for (final prop in props) {
    if (o.containsKey(prop)) {
      result[prop] = o[prop];
    }
  }
  return result;
}

/// Calculate the length of a string, taking multi-byte characters into account.
/// This mimics the behavior of Python's `len` function.
/// @param {string} s The string to calculate the length of.
/// @returns {number} The length of the string.
int len(String s) => s.runes.length;
