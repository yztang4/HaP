import torch
import torch.nn as nn
import pytorch3d.ops
import numpy as np

class CLGD(nn.Module):
    # assign weight for each query
    def __init__(self,up_ratio=10,K=5,std_factor=3,weighted_query=False,beta=0):
        '''
        Symbol in the manuscript:
        up_ratio is R
        K is K
        std_factor is T
        weighted_query=True means beta>0, and vice versa
        beta is beta
        '''
        super(CLGD,self).__init__()
        self.K=K
        self.up_ratio=up_ratio
        self.std_factor=std_factor
        self.weighted_query=weighted_query
        self.beta=beta

    def cal_udf_weights(self,x,query):
        #x: (B,N,3)
        #query=self.grid_flatten.to(x).unsqueeze(0).repeat(x.size(0),1,1)
        
        dists,idx,knn_pc=pytorch3d.ops.knn_points(query,x,K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)

        dir=query.unsqueeze(2)-knn_pc   #(B,N,K,3)

        #weights=torch.softmax(-dists.sqrt(),dim=2)   #(B,N,K) weight more, dist small
        #weights=torch.softmax(-dists,dim=2)   #(B,N,K) weight more, dist small
        #weights=torch.softmax(-dists/torch.min(dists,dim=2,keepdim=True)[0],dim=2)   #(B,N,K) weight more, dist small

        norm = torch.sum(1.0 / (dists + 1e-8), dim = 2, keepdim = True)
        weights = (1.0 / (dists.detach() + 1e-8)) / norm.detach()


        #print(weights)
        #assert False

        udf=torch.sum((dists+1e-10).sqrt()*weights,dim=2)  #(B,N)
        #udf=torch.sum(dists*weights,dim=2)  #(B,N)

        udf_grad=torch.sum(dir*weights.unsqueeze(-1),dim=2) #(B,N,3)

        return udf,udf_grad,weights

    def cal_udf(self,x,weights,query):
        #query=self.grid_flatten.to(x).unsqueeze(0).repeat(x.size(0),1,1)

        dists,idx,knn_pc=pytorch3d.ops.knn_points(query,x,K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)
        dir=query.unsqueeze(2)-knn_pc   #(B,N,K,3)
        udf=torch.sum((dists+1e-10).sqrt()*weights,dim=2)  #(B,N)
        #udf=torch.sum(dists*weights,dim=2)  #(B,N)

        udf_grad=torch.sum(dir*weights.unsqueeze(-1),dim=2) #(B,N,3)
        return udf,udf_grad

    def forward(self,src,tgt):
        #src: target (B,N,3)
        #tgt: source (B,N,3)

        with torch.no_grad():
            # !!! fix !!! return_sorted to make dists sorted !!!
            tgt_self_dists,_,_=pytorch3d.ops.knn_points(tgt,tgt,return_nn=True,K=2,return_sorted=True)
            
            tgt_self_dists=tgt_self_dists[:,:,1:]   #(B,N,1)
            tgt_self_dists=torch.sqrt(tgt_self_dists+1e-10)

            std=tgt_self_dists*self.std_factor


            noise_offset=torch.randn(tgt.size(0),tgt.size(1),self.up_ratio,3).to(tgt).float() * std.unsqueeze(3)

            #query=query_center.unsqueeze(2)+noise_offset
            query=tgt.unsqueeze(2)+noise_offset
            query=query.reshape(tgt.size(0),-1,3).detach()

        #for i in range(self.up_ratio):
        query=torch.cat((query,src.detach()),dim=1)

        '''if not os.path.exists('query.xyz'): #for each gt, we have to make a query point cloud first
            np.savetxt('query.xyz',query.cpu().detach().numpy().squeeze(0))
        '''

        udf_tgt,udf_grad_tgt,weights=self.cal_udf_weights(tgt,query)
        
        udf_src,udf_grad_src=self.cal_udf(src,weights,query)

        udf_error=torch.abs(udf_tgt-udf_src)    #(B,M)
        #udf_loss=torch.mean(torch.square(udf_tgt-udf_src))

        #udf_grad_loss=torch.mean(1-torch.sum(udf_grad_src*udf_grad_tgt,dim=-1))

        
        udf_grad_error=torch.sum(torch.abs(udf_grad_src-udf_grad_tgt),axis=-1)  #(B,M)
        #udf_grad_loss=torch.mean(torch.square(udf_grad_src-udf_grad_tgt))

        if self.weighted_query:

            with torch.no_grad():
                query_weights=torch.exp(-udf_error.detach()*self.beta)*torch.exp(-udf_grad_error.detach()*self.beta)
            return torch.sum((udf_error+udf_grad_error)*query_weights.detach())/query.size(0)/query.size(1)
        
        else:
            query_weights=1
            return torch.sum((udf_error+udf_grad_error)*query_weights)/query.size(0)/query.size(1)