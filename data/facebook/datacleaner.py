seq_in=input.split("\n")
seq_out=output.split("\n")

counter=0
outfile_in = open('train_wrong_in.txt','w')
outfile_out = open('train_wrong_out.txt','w')
assert(len(seq_in)==len(seq_out))
for i in range(len(seq_in)):
    if len(seq_in[i].split())!=len(seq_out[i].split()):
        counter+=1
        outfile_in.write(seq_in[i] + "\n")
        outfile_out.write(str(i+1) + " " + seq_out[i] + "\n")

outfile_in.close()
outfile_out.close()

print(counter)
print("finished")