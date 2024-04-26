import argparse, os, shutil
import sys
from exl2conv.conversion.qparams import qparams_headoptions
from exl2conv.conversion.convert import convert_hf_to_exl2
import torch

def main():
    parser = argparse.ArgumentParser(description = "Convert model to ExLlamaV2")
    parser.add_argument("-i", "--in_dir", type = str, help = "Input directory", default = "")
    parser.add_argument("-o", "--out_dir", type = str, help = "Output (working) directory")
    parser.add_argument("-nr", "--no_resume", action = "store_true", help = "Do not resume an interrupted job (deletes all files in the output directory)")
    parser.add_argument("-cf", "--compile_full", type = str, help = "Output folder for compiled model with all config/tokenizer files")
    parser.add_argument("-c", "--cal_dataset", type = str, help = "Calibration dataset (.parquet file)")
    parser.add_argument("-b", "--bits", type = float, default = 4.125, help = "Target bits per weight")
    parser.add_argument("-ss", "--shard_size", type = float, help = "Max shard size in MB (default: 8192)", default = 8192)
    parser.add_argument("-rs", "--rope_scale", type = float, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, help = "RoPE alpha value (NTK)")
    parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Target bits per weight (head layer)")
    parser.add_argument("-om", "--output_measurement", type = str, help = "Only perform measurement pass, then save measurement to the specified file")
    parser.add_argument("-m", "--measurement", type = str, help = "Reuse previous measurement")
    parser.add_argument("-r", "--dataset_rows", type = int, default = 100, help = "Number of rows to apply from dataset")
    parser.add_argument("-mr", "--measurement_rows", type = int, default = 16, help = "Number of rows to apply from dataset when measuring")
    parser.add_argument("-l", "--length", type = int, default = 2048, help = "Max no. tokens per sample")
    parser.add_argument("-ml", "--measurement_length", type = int, default = 2048, help = "Max no. tokens per sample when measuring")

    args = parser.parse_args()

    torch.set_printoptions(precision = 7, sci_mode = False, linewidth = 200)

    # Check some args

    if not args.in_dir:
        print(" ## Please specify input model directory (-i, --in_dir)")
        sys.exit()

    if not args.out_dir:
        print(" ## Please specify output/working directory (-o, --out_dir)")
        sys.exit()

    if args.length > 2048 or args.measurement_length > 2048:
        print(" !! Warning: calibration rows > 2048 tokens may result in excessive VRAM use")

    if not args.head_bits in qparams_headoptions:
        print(f" ## Error: {args.head_bits} is not a supported option for head layer bitrate")
        sys.exit()

    if args.output_measurement is not None and args.compile_full is not None:
        print(" ## Conflicting options: --output_measurement and --compile_full")
        sys.exit()

    if args.bits < 2 or args.bits > 8:
        print(f" !! Warning: target bitrate {args.bits} will likely not be attainable")

    if not os.path.exists(args.out_dir):
        print(f" ## Error: Directory not found: {args.out_dir}")
        sys.exit()

    job = {
        "in_dir": args.in_dir,
        "out_dir": args.out_dir,
        "cal_dataset": args.cal_dataset,
        "bits": args.bits,
        "dataset_rows": args.dataset_rows,
        "measurement_rows": args.measurement_rows,
        "length": args.length,
        "measurement_length": args.measurement_length,
        "measurement": args.measurement,
        "head_bits": args.head_bits,
        "shard_size": args.shard_size if args.shard_size > 0 else 1024 ** 3,  # 1 PB = unlimited,
        "compile_full": args.compile_full,
        "rope_scale": args.rope_scale,
        "rope_alpha": args.rope_alpha,
        "no_resume": args.no_resume,
        "output_measurement": args.output_measurement,
    }

    convert_hf_to_exl2(job)

if __name__ == "__main__":
    main()
