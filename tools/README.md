Convert tfrecords files matching the given file pattern to npz files with
  ```
  python ConvertToNumpy.py [options] --data=[file pattern]
  ```
    
Convert npz files matching the given file pattern to tfrecord files with
  ```
  python ConvertToTfrecord.py [options] --data=[file pattern]
  ```
    
Both tools currently only work with the tfrecord (npz) file patterns specified in simulations/ConvertToTfrecord (ConvertToNumpy)
