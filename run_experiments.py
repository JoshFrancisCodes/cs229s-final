import subprocess
import json

def run_training(batch_size):
  result = subprocess.run(['python', 'train.py', f'--batch_size={batch_size}'], capture_output=True, text=True)
  return result.stdout

def process_output(output):
    # Split the output into lines
    lines = output.strip().split('\n')

    # Get the last line
    last_line = lines[-1]

    # Extract the loss and throughput values
    # Assuming the format is exactly "final loss: <loss_value>, final throughput: <throughput_value>"
    parts = last_line.split(',')
    loss_str = parts[0].split(': ')[1]
    throughput_str = parts[1].split(': ')[1]

    # Convert string values to float
    loss = float(loss_str.strip())
    throughput = float(throughput_str.strip())

    return loss, throughput

def main():
  results = {"loss": None, "training_throughput_4": None, "training_throughput_8": None}

  output_4 = run_training(4)
  loss_4, throughput_4 = process_output(output_4)
  results["loss"] = loss_4  # Assuming loss is the same for both
  results["training_throughput_4"] = throughput_4
  print("RESULTS: ", results)

  # Run training and process output for batch size 8
  output_8 = run_training(8)
  _, throughput_8 = process_output(output_8)
  results["training_throughput_8"] = throughput_8
  print("RESULTS: ", results)

  with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
  

if __name__ == '__main__':
  main()