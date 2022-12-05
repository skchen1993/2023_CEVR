import json

path = 'runs/large/test_result.json'

with open(path, 'r') as fp:
    result = json.load(fp)

psnr_list = result[0:3] + result[4:]

#print( result, psnr_list)

print(sum(psnr_list) /6)