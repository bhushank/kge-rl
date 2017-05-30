
import os

def main():
    data_set = 'freebase'
    base = '/home/mitarb/kotnis/Data/neg_sampling/{}/'.format(data_set)
    models={'rescal','transE','distmult','complex'}
    samplers = {'corrupt','random','relational','typed','adversarial','nn'}
    nums = [1,2,5,10,20,50,100]
    results=dict.fromkeys(samplers,0)
    for sampler in samplers:
        results[sampler] = dict()
        for model in models:
            results[sampler][model] =dict()
            for n in nums:
                results[sampler][model][n] = dict()

    for sampler in samplers:
        for model in models:
            for n in nums:
                file_path = os.path.join(base,sampler,model.lower(),model+"_"+str(n),'results_test_filt')
                rank_path = os.path.join(base,sampler,model.lower(),model+"_"+str(n),'ranks_test_filt')
                if os.path.exists(file_path):
                    mrr,hits_10 = read_file(file_path,rank_path,True)
                    #print("{},{},{}".format(sampler,model,n))
                    #print("{},{}".format(n, mrr))
                    results[sampler][model][n]['mrr'] = mrr
                    results[sampler][model][n]['hits10'] = hits_10

                else:
                    print("{} does not exist!".format(file_path))
    #print results
    write_dict(results,data_set)

def write_dict(results,data_set):
    nums = [1, 2, 5, 10, 20, 50, 100]
    base = '/home/mitarb/kotnis/Data/neg_sampling/{}/results/'.format(data_set)
    for sampler in results:
        res_dir = base + 'results_{}'.format(sampler)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        for model in results[sampler]:
            with open(res_dir+"/"+'{}.csv'.format(model),'w') as f:
                for n in nums:
                    f.write("{},{}\n".format(results[sampler][model][n].get('mrr',0.0),results[sampler][model][n].get('hits10',0.0)))


def read_file(file_path,rank_path='', is_h1=False):
    res = []
    with open(file_path) as r:
        for line in r:
            _,val = line.strip().split(" : ")
            res.append(float(val))
    if is_h1:
        num_h1 = 0
        num_count = 0
        with open(rank_path) as f:
            for line in f:
                line = line.strip()
                rank = float(line)
                if rank<= 10.0:
                    num_h1 += 1
                num_count += 1
        h1 = num_h1/float(num_count)
        res[-1] = h1

    return res

if __name__=='__main__':
    main()