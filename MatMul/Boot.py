import subprocess as subproc

def build_matmul(build_path):
  subproc.run(['cd',  build_path, '&&', 'make'], check=True)

def run_matmul(matmul_path, mode, parameters):
  result = []
  for size in parameters:
    time_output = subproc.run([matmul_path, 
                                  mode,
                                  '--matrix',
                                  '2048',
                                  '2048',
                                  str(size),
                                  '--print-only-time'], stdout=subproc.PIPE)
    time = 0
    try:
        time = int(time_output.stdout) 
    except ValueError:
        print(time_output.stdout, 'is not an int')
        return
    result.append(time)
  return result
