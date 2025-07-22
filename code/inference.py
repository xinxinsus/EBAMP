import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--token_path", type=str, default="./my_token")
parser.add_argument("--model_path", type=str, default="./my_model_fintune")
parser.add_argument("--max_length", type=int, default=45)
parser.add_argument("--min_length", type=int, default=16)
parser.add_argument("--num_sequences", type=int, default=50000)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument("--special_begin", type=str, default="LAMPXXX:")
parser.add_argument("--devices", type=str, default="0")

args = parser.parse_args()

# set the device
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices


# load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.token_path)
# load the model
model = GPT2LMHeadModel.from_pretrained(args.model_path)
# set the model to eval mode
model.eval()
# set the model to cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)


outputs = []
# start to generate sequences
seed = 0
while len(outputs)<args.num_sequences:
    # set the seed
    torch.manual_seed(seed)
    seed = seed +1
    # generate the sequence
    sequence = model.generate(
        input_ids=torch.tensor(tokenizer.encode(f"?{args.special_begin}")).unsqueeze(0).to(device),
        max_length=args.max_length,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        do_sample=True,
        top_k=0,
        top_p=0.9,
        num_return_sequences=1,
    )
    # decode the sequence
    sequence = tokenizer.decode(sequence[0], skip_special_tokens=True)
    # print the sequence
    if sequence in outputs or len(sequence)<args.min_length or len(sequence)>args.max_length or sequence[len(sequence)-1]!= '!':
        continue
    outputs.append(sequence)
    print(sequence)
    if len(outputs)%100 == 0:
        print('***************'+str(len(outputs))+'***************')

# save the sequences
with open('./data/predict_{}.fasta'.format(args.num_sequences), "w") as f:
    idx = 1
    for output in outputs:
        seq = output.split(args.special_begin)[1][:-1]
        f.write(">"+str(idx) + "\n")
        f.write(seq + "\n")
        idx = idx + 1
