import unittest
import numpy as np
from align_images import rigid, fit_labels


class AlignmentTest(unittest.TestCase):
    def test_fixed_scale_rigid_transform(self):
        p=np.array([[0.,0.],[100,0],[0,200],[150,170]])
        angle=.13
        expected=.7*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        shift=np.array([23.,-41.])
        matrix,t=rigid(p,p@expected.T+shift,.7)
        np.testing.assert_allclose(matrix,expected,atol=1e-12)
        np.testing.assert_allclose(t,shift,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(matrix),.7**2)

    def test_wrong_label_is_rejected(self):
        pixels=np.array([[100,100],[500,120],[120,500],[520,520],[800,800]],float)
        p=(pixels-[500,500])*[1,-1]
        angle=-.04
        matrix=.683*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        q=p@matrix.T+[100,200]
        q[-1]+=[400,600]
        lib=[{'center':v} for v in q]
        detections=[{'index':i,'pixel':v.tolist()} for i,v in enumerate(pixels)]
        recovered,t,good=fit_labels(detections,lib,(1000,1000),.683)
        self.assertEqual(good.tolist(),[True,True,True,True,False])
        np.testing.assert_allclose(recovered,matrix,atol=1e-12)
        np.testing.assert_allclose(t,[100,200],atol=1e-12)


if __name__=='__main__':unittest.main()
