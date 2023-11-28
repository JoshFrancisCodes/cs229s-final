import subprocess
import json

def run_training(batch_size):
  result = subprocess.run(['python', 'train.py', f'--batch_size={batch_size}'], capture_output=True, text=True)
  return result.stdout

def run_sampling(num_samples, new_tokens, batch_size):
  result = subprocess.run(['python', 'sample.py', '--init_from=gpt2', f'--num_samples={num_samples}', f'--max_new_tokens={new_tokens}', f'--batch_size={batch_size}'], capture_output=True, text=True)
  return result.stdout

def process_train_output(output):
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

def process_sample_output(output):
    # Split the output into lines
    lines = output.strip().split('\n')

    # Get the last line
    last_line = lines[-1]

    # Extract the loss and throughput values
    # Assuming the format is exactly "final loss: <loss_value>, final throughput: <throughput_value>"
    parts = last_line.split(':')

    # Convert string values to float
    tokens_per_sec = float(parts[1].strip())

    return tokens_per_sec

def main():
  results = {"loss": None, "training_throughput_4": None, "training_throughput_8": None, "inference_latency_1": None, "inference_latency_12": None}

  # train_output_4 = run_training(4)
  # loss_4, throughput_4 = process_train_output(train_output_4)
  # results["loss"] = loss_4  # Assuming loss is the same for both
  # results["training_throughput_4"] = throughput_4
  # print("RESULTS: ", results)

  # Run training and process output for batch size 8
  # train_output_8 = run_training(8)
  # _, throughput_8 = process_train_output(train_output_8)
  # results["training_throughput_8"] = throughput_8
  # print("RESULTS: ", results)

  sample_output_1 = run_sampling(50, 1024, 1)
  tokens_per_sec_1 = process_sample_output(sample_output_1)
  results["inference_latency_1"] = tokens_per_sec_1

  sample_output_12 = run_sampling(50, 1024, 12)
  tokens_per_sec_12 = process_sample_output(sample_output_12)
  results["inference_latency_12"] = tokens_per_sec_12



  with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=4)
  

if __name__ == '__main__':
  main()
