import argparse


parser = argparse.ArgumentParser(description="adding and deleting stuff")
parser.add_argument("num1", help="Number 1", type=int)
parser.add_argument("num2", help="Number 2", type=int)
parser.add_argument("-a", "--action", help="What action do you want to take on the inserted numbers", default="+", choices=["+", "-"])
parser.add_argument("testing", action=VerboseStore)
args = parser.parse_args()

def math(args):
    if args.action == "+":
        value = args.num1 + args.num2
        return value
    if args.action == "-":
        value = args.num1 - args.num2
        return value
    else:
        return "Incorrect --action value"

print(math(args))
