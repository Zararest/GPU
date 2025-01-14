#!/usr/bin/python3

import subprocess
import argparse
import matplotlib.pyplot as plt 
import matplotlib

def generate_graph_data(nodes_num, out_file):
    # Run Graph-generator.py to generate graph data
    file_name = f"./build/{nodes_num}-{out_file}"
    subprocess.run(['./build/Graph-generator', '--node-size', '3' ,'--nodes-num', str(nodes_num), 
                    '--out-file', file_name])
    return file_name

def measure_performance(in_file, use_gpu):
    # Run Perf-measure with the given input file and mode
    print(f"Performance on file {in_file}")
    mode = '--use-GPU' if use_gpu else '--use-CPU'
    result = subprocess.run(['./build/Perf-measure', '--only-time', mode, '--in-file', in_file], 
                            capture_output=True, text=True)
    time_ms = float(result.stdout)  # Convert time to milliseconds
    return time_ms

def main():
  matplotlib.rcParams.update({'font.size': 16})
  parser = argparse.ArgumentParser(description="Measure performance of Perf-measure in different modes")
  parser.add_argument('--max-nodes', type=int, default=100, help="Maximum number of nodes")
  args = parser.parse_args()

  out_files = []
  first_measure = 2
  # Generate graph data
  for nodes_num in range(first_measure, args.max_nodes + 1):
    out_files.append(generate_graph_data(nodes_num, 'tmp.out'))

  cpu_times = []
  gpu_times = []

  CPU_too_slow = False
  # Measure performance for each mode
  for nodes_num in range(first_measure, args.max_nodes + 1):
    if not CPU_too_slow:
      time_cpu = measure_performance(out_files[nodes_num - first_measure], use_gpu=False)
      time_gpu = measure_performance(out_files[nodes_num - first_measure], use_gpu=True)
      cpu_times.append(time_cpu)
      gpu_times.append(time_gpu)
      CPU_too_slow = time_cpu / time_gpu > 70
    else:
      time_gpu = measure_performance(out_files[nodes_num - first_measure], use_gpu=True)
      gpu_times.append(time_gpu)

  # Create the graph
  plt.plot(range(2, args.max_nodes + 1)[0:len(gpu_times)], gpu_times)
  plt.scatter(range(2, args.max_nodes + 1)[0:len(gpu_times)], gpu_times, marker='*',
              label='GPU')

  plt.plot(range(2, args.max_nodes + 1)[0:len(cpu_times)], cpu_times)
  plt.scatter(range(2, args.max_nodes + 1)[0:len(cpu_times)], cpu_times, label='CPU', 
              marker='o',)
  plt.xlabel('Количество узлов в графе')
  plt.ylabel('Время поиска[мс]')
  plt.title('')
  plt.legend()
  plt.grid()
  plt.show()

if __name__ == "__main__":
    main()
