"""Mechanism checks against independent small-data oracles, not score tuning."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from train import cosine_weights,disjoint_batches,oversample,LR,fed_round

class MechanismTests(unittest.TestCase):
    def test_lha_cosine_and_zero_vector(self):
        clients={'a':{'x':np.array([[1,0],[0,1]],dtype='float32')},'b':{'x':np.array([[1,1]],dtype='float32')}}
        public=np.array([[1,0],[0,0]],dtype='float32');actual=cosine_weights(clients,public).numpy()
        a=.5;b=1/np.sqrt(2);np.testing.assert_allclose(actual[:,0],[a/(a+b),b/(a+b)],rtol=1e-6)
        np.testing.assert_array_equal(actual[:,1],0)

    def test_tiny_client_disjoint_support_query(self):
        project=SimpleNamespace(x_train=np.arange(6,dtype='float32').reshape(3,2),y_train=np.array([0,1,0]))
        sx,_,qx,_=disjoint_batches(project,{'training':{'batch_size':32}},np.random.default_rng(1),torch.device('cpu'))
        self.assertFalse(set(sx[:,0].tolist())&set(qx[:,0].tolist()))

    def test_ros_only_duplicates_existing_rows(self):
        x=np.arange(12,dtype='float32').reshape(6,2);y=np.array([0,0,0,0,0,1])
        xx,yy=oversample(x,y,np.random.default_rng(0));self.assertEqual(int(yy.sum()),5);self.assertEqual(len(xx),10)
        self.assertTrue(set(xx[:,0].tolist())<=set(x[:,0]))

    def test_kd_uses_teacher_weighted_logits_and_updates(self):
        torch.set_num_threads(1);torch.manual_seed(2);rng=np.random.default_rng(2)
        x=np.array([[1,0],[0,1],[1,1],[.1,.2]],dtype='float32');y=np.array([0,1,0,1])
        clients={n:dict(x=x,y=y,sampled=(torch.from_numpy(x),torch.from_numpy(y))) for n in ['a','b']}
        # Author aggregation counts the post-oversampling dataset, even though
        # LHA continues to use the original four training rows.
        clients['b']['sampled']=(torch.from_numpy(np.concatenate([x,x[:2]])),torch.from_numpy(np.concatenate([y,y[:2]])))
        model=LR(2);before=model.linear.weight.detach().clone();alpha=cosine_weights(clients,x)
        cfg=dict(public_sample_limit=700,local_lr=.001,local_epochs=2,batch_size=2,distill_lr=.0001,distill_epochs=2,temperature=5)
        record=fed_round(model,clients,x,alpha,cfg,rng)
        self.assertFalse(torch.equal(before,model.linear.weight));self.assertEqual(len(record['public_indices']),4)
        self.assertTrue(np.isfinite(record['kd_loss']))
        self.assertEqual(record['aggregation_sample_counts'],{'a':4,'b':6})

if __name__=='__main__':unittest.main()
