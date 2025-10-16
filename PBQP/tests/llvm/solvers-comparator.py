import os
import re
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SolverComparator:
    def __init__(self, cpp_executable: str = "./solver"):
        """
        Initialize the SolverComparator.
        
        Args:
            cpp_executable: Path to the C++ solver executable
        """
        self.cpp_executable = cpp_executable
        self.results = []
        
    def find_file_pairs(self, directory: str = ".") -> List[Tuple[str, str, str]]:
        """
        Find all pairs of .pbqpgraph and .solution files.
        
        Returns:
            List of tuples (base_name, graph_file, solution_file)
        """
        directory_path = Path(directory)
        graph_files = list(directory_path.glob("*.pbqpgraph"))
        file_pairs = []
        
        for graph_file in graph_files:
            base_name = graph_file.stem  # Remove .pbqpgraph extension
            solution_file = graph_file.with_suffix('.solution')
            
            if solution_file.exists():
                file_pairs.append((base_name, str(graph_file), str(solution_file)))
            else:
                print(f"Warning: No solution file found for {graph_file}")
                
        return file_pairs
    
    def parse_solution_file(self, solution_file: str) -> Optional[Tuple[float, float]]:
        """
        Parse .solution file and extract Total cost and Execution time.
        
        Returns:
            Tuple of (total_cost, execution_time) or None if parsing fails
        """
        try:
            with open(solution_file, 'r') as f:
                content = f.read()
            
            # Extract Total cost
            cost_match = re.search(r"Total cost:\s*([\d.]+)", content)
            # Extract Execution time
            time_match = re.search(r"Execution time:\s*([\d.]+)", content)
            
            if cost_match and time_match:
                total_cost = float(cost_match.group(1))
                execution_time = float(time_match.group(1))
                return total_cost, execution_time
            else:
                print(f"Warning: Could not parse solution file {solution_file}")
                return None
                
        except Exception as e:
            print(f"Error reading solution file {solution_file}: {e}")
            return None
    
    def parse_gpu_solution_file(self, gpu_solution_file: str) -> Optional[Tuple[float, float, float]]:
        """
        Parse .solution-gpu file and extract Total cost, Execution time, and Pure execution time.
        
        Returns:
            Tuple of (total_cost, execution_time, pure_execution_time) or None if parsing fails
        """
        try:
            with open(gpu_solution_file, 'r') as f:
                content = f.read()
            
            # Extract Total cost
            cost_match = re.search(r"Total cost:\s*([\d.]+)", content)
            # Extract Execution time
            time_match = re.search(r"Execution time:\s*([\d.]+)", content)
            # Extract Pure execution time
            pure_time_match = re.search(r"Pure execution time:\s*([\d.]+)", content)
            
            if cost_match and time_match and pure_time_match:
                total_cost = float(cost_match.group(1))
                execution_time = float(time_match.group(1))
                pure_execution_time = float(pure_time_match.group(1))
                return total_cost, execution_time, pure_execution_time
            else:
                print(f"Warning: Could not parse GPU solution file {gpu_solution_file}")
                return None
                
        except Exception as e:
            print(f"Error reading GPU solution file {gpu_solution_file}: {e}")
            return None
    
    def run_cpp_solver(self, graph_file: str) -> bool:
        """
        Run C++ solver on the given graph file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.cpp_executable):
                print(f"Error: C++ executable not found at {self.cpp_executable}")
                return False
            
            result = subprocess.run(
                [self.cpp_executable, graph_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Warning: C++ solver returned non-zero exit code for {graph_file}")
                print(f"Stderr: {result.stderr}")
                return False
                
            return True
            
        except subprocess.TimeoutExpired:
            print(f"Error: C++ solver timed out for {graph_file}")
            return False
        except Exception as e:
            print(f"Error running C++ solver for {graph_file}: {e}")
            return False
    
    def process_single_file(self, base_name: str, graph_file: str, solution_file: str) -> Optional[Dict]:
        """
        Process a single file pair and collect data.
        
        Returns:
            Dictionary with collected data or None if processing fails
        """
        print(f"Processing {base_name}...")
        
        # Parse CPU solution file
        cpu_data = self.parse_solution_file(solution_file)
        if cpu_data is None:
            return None
        
        cpu_cost, cpu_time = cpu_data
        
        # Run C++ solver to generate GPU solution
        if not self.run_cpp_solver(graph_file):
            return None
        
        # Parse GPU solution file
        gpu_solution_file = graph_file + "-gpu"
        if not os.path.exists(gpu_solution_file):
            print(f"Warning: GPU solution file not found at {gpu_solution_file}")
            return None
        
        gpu_data = self.parse_gpu_solution_file(gpu_solution_file)
        if gpu_data is None:
            return None
        
        gpu_cost, gpu_time, gpu_pure_time = gpu_data
        
        return {
            'name': base_name,
            'cpu_total_cost': cpu_cost,
            'cpu_execution_time': cpu_time,
            'gpu_total_cost': gpu_cost,
            'gpu_execution_time': gpu_time,
            'gpu_pure_execution_time': gpu_pure_time
        }
    
    def process_all_files(self, directory: str = ".") -> None:
        """Process all file pairs in the given directory."""
        file_pairs = self.find_file_pairs(directory)
        
        if not file_pairs:
            print("No file pairs found!")
            return
        
        print(f"Found {len(file_pairs)} file pairs to process")
        
        for base_name, graph_file, solution_file in file_pairs:
            result = self.process_single_file(base_name, graph_file, solution_file)
            if result is not None:
                self.results.append(result)
                print(f"Successfully processed {base_name}")
            else:
                print(f"Failed to process {base_name}")
    
    def generate_comparison_table(self, output_file: str = "solver_comparison.csv") -> None:
        """Generate a CSV file with the comparison results."""
        if not self.results:
            print("No results to export!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Add comparison columns
        df['cost_difference'] = df['gpu_total_cost'] - df['cpu_total_cost']
        df['cost_ratio'] = df['gpu_total_cost'] / df['cpu_total_cost']
        df['time_speedup'] = df['cpu_execution_time'] / df['gpu_execution_time']
        df['pure_time_speedup'] = df['cpu_execution_time'] / df['gpu_pure_execution_time']
        
        # Reorder columns for better readability
        columns = [
            'name',
            'cpu_total_cost', 'gpu_total_cost', 'cost_difference', 'cost_ratio',
            'cpu_execution_time', 'gpu_execution_time', 'gpu_pure_execution_time',
            'time_speedup', 'pure_time_speedup'
        ]
        df = df[columns]
        
        df.to_csv(output_file, index=False)
        print(f"Comparison table saved to {output_file}")
        
        # Print summary statistics
        self.print_summary_statistics(df)
    
    def print_summary_statistics(self, df: pd.DataFrame) -> None:
        """Print summary statistics of the comparison."""
        print("\n=== Summary Statistics ===")
        print(f"Total files processed: {len(df)}")
        print(f"Average cost ratio (GPU/CPU): {df['cost_ratio'].mean():.4f}")
        print(f"Average time speedup: {df['time_speedup'].mean():.4f}")
        print(f"Average pure time speedup: {df['pure_time_speedup'].mean():.4f}")
        print(f"Best time speedup: {df['time_speedup'].max():.4f}")
        print(f"Worst time speedup: {df['time_speedup'].min():.4f}")

def main():
    """Main function to run the solver comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare CPU and GPU solvers')
    parser.add_argument('--cpp-executable', default='./solver',
                       help='Path to C++ solver executable (default: ./solver)')
    parser.add_argument('--directory', default='.',
                       help='Directory containing the files (default: current directory)')
    parser.add_argument('--output', default='solver_comparison.csv',
                       help='Output CSV file name (default: solver_comparison.csv)')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = SolverComparator(cpp_executable=args.cpp_executable)
    
    # Process all files
    comparator.process_all_files(directory=args.directory)
    
    # Generate comparison table
    comparator.generate_comparison_table(output_file=args.output)

if __name__ == "__main__":
    main()