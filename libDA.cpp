// Compile with cmake (CMakeLists.txt is provided) or with the following lines in bash:
// g++ -c -fPIC libautosim.cpp -o libautosim.o
// g++ -shared -Wl,-soname,libautosim.so -o libautosim.so libautosim.o


#include <string>
#include <sstream>
#include <iostream>
#include <list>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <float.h>
#include <limits>
#include <cmath>
#include <limits>

#ifdef _USAC
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include "ConfigParams.h"
#include "cpp_libs/libUSAC/estimators/FundmatrixEstimator.h"
#include "cpp_libs/libUSAC/estimators/HomogEstimator.h"
#include "cpp_libs/libUSAC/estimators/FundmatrixEstimator.h"
#endif


#include <bitset>
#include "cpp_libs/libOrsa/orsa_fundamental.hpp"
#include "cpp_libs/libOrsa/orsa_homography.hpp"

#include<cpp_libs/ACmatch.h>


#define FullDescDim 6272
#define VecDescDim 128 // as in (x,x,128) output of the network
#define SameKPThres 4


struct TargetNode
{
  const int TargetIdx; float sim_with_query;
  TargetNode(int Idx, float sim):TargetIdx(Idx){ this->sim_with_query = sim; };
  bool operator >(const TargetNode & kp) {return ( this->sim_with_query>kp.sim_with_query );};
  bool operator ==(const TargetNode & kp) {return ( this->sim_with_query==kp.sim_with_query );};
};

struct QueryNode
{
  const int QueryIdx;
  std::list<QueryNode>::iterator thisQueryNodeOnList; //pointer on list
  float first_sim, last_sim;
  std::list<TargetNode> MostSimilar_TargetNodes;

  void Add_TargetNode(int it, float sim, int MaxTnodes_num)
  {
    TargetNode tn(it,sim);
    std::list<TargetNode>::iterator target_iter;
    for(target_iter = MostSimilar_TargetNodes.begin(); target_iter != MostSimilar_TargetNodes.end(); ++target_iter)
      if ( tn > *target_iter )
        break;

    MostSimilar_TargetNodes.insert( target_iter, tn );
    if (MaxTnodes_num>0 && MostSimilar_TargetNodes.size()>MaxTnodes_num)
      MostSimilar_TargetNodes.pop_back();
      last_sim = (--MostSimilar_TargetNodes.end())->sim_with_query;
    first_sim = MostSimilar_TargetNodes.begin()->sim_with_query;  
  };

  QueryNode(int Idx):QueryIdx(Idx){ first_sim = -1; last_sim = -1; };
};

struct DescStatsClass
{
  float norm, max, min, mean, sigma;
  std::bitset<FullDescDim> AID;
  std::bitset<VecDescDim> AIDbyVec [FullDescDim/VecDescDim];

  DescStatsClass()
  {
    norm = 0.0f, mean = 0.0f, sigma = 0.0f;
    max = -std::numeric_limits<float>::infinity();
    min = std::numeric_limits<float>::infinity();
  };
};


int CountItFast(const DescStatsClass & q, const DescStatsClass & t, int thres)
{
  int xor_opp = 0;
  for (int v = 0; (v < FullDescDim/VecDescDim && (xor_opp < thres)); v++)
    xor_opp +=  (q.AIDbyVec[v] ^ t.AIDbyVec[v]).count();
  return(xor_opp);
}


float FastSimi(int iq, float* DescsQuery, DescStatsClass* QueryStats, int it, float* DescsTarget, DescStatsClass* TargetStats, float simi_thres)
{
  float m_sq = QueryStats[iq].max * TargetStats[it].max;
  float norms_prod = QueryStats[iq].norm * TargetStats[it].norm;
  float dynamic_thres = (simi_thres * norms_prod) - FullDescDim * m_sq;
  float SP = 0.0;
  int qpos = iq*FullDescDim, tpos = it*FullDescDim;
  for (int i = 0; (i < FullDescDim); i++)
  {
    SP +=  DescsQuery[qpos + i] * DescsTarget[tpos + i];
  }
  return(SP/norms_prod);
};

DescStatsClass LoadInfo(int iq, float* DescsQuery)
{
  DescStatsClass ds;
  float val;
  int qpos = iq*FullDescDim;
  for (int i = 0; (i < FullDescDim); i++)
  {
    val = DescsQuery[qpos + i];
    ds.norm += val * val;
  }
  
  for (int i = 0; (i < FullDescDim); i++)
    if (DescsQuery[qpos + i]>=0)
    {
      ds.AID.set(i);
    }

  for (int v = 0; (v < FullDescDim/VecDescDim); v++)
  {
    int vpos = VecDescDim*v;
    for (int i = 0; (i < VecDescDim); i++)
      {
        if (DescsQuery[qpos + vpos + i]>=0)
          ds.AIDbyVec[v].set(i);
      }
  }

  ds.norm = std::sqrt(ds.norm);
  return(ds);
};


struct KeypointClass{
  float x,y;
};


struct MatchersClass
{
  std::list<QueryNode> QueryNodes;
  const int k; // k as in knn, if <=0 then we store all matches having a sim above sim_thres
  const float sim_thres;
  const int Nvec;
  std::vector<KeypointClass> QueryKPs, TargetKPs;
  
  int* FilteredIdxMatches;
  int N_FilteredMatches = 0;

  float* vlfeatData;
  int N_vlfeatData=0;
  int vl_patchWidth=0;
  unsigned char * vlfeatPatches;
  int vl_desc_dim=0;
  float* vl_desc;
  



  MatchersClass(int knn_num, float sim_thres) : k(knn_num), sim_thres(sim_thres), Nvec((int)(FullDescDim / VecDescDim)){};

  void KnnMatcher(float *DescsQuery, int Nquery, float *DescsTarget, int Ntarget, int FastCode)
  {
    DescStatsClass *QueryStats = new DescStatsClass[Nquery], *TargetStats = new DescStatsClass[Ntarget];
#pragma omp parallel for default(shared)
    for (int iq = 0; iq < Nquery; iq++)
      QueryStats[iq] = LoadInfo(iq, DescsQuery);
#pragma omp parallel for default(shared)
    for (int it = 0; it < Ntarget; it++)
      TargetStats[it] = LoadInfo(it, DescsTarget);

    switch (FastCode)
    {
      case 0:
      { // BigAID
        // std::cout<<"---> Brute Force Angle Comparisons"<<std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          for (int it = 0; it < Ntarget; it++)
          {
            float updated_sim_thres = (this->k > 0 && qn.last_sim > sim_thres) ? qn.last_sim : sim_thres;
            float simi = FastSimi(iq, DescsQuery, QueryStats, it, DescsTarget, TargetStats, updated_sim_thres);
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          if (qn.first_sim > sim_thres)
  #pragma omp critical
          {
            QueryNodes.push_back(qn);
            std::list<QueryNode>::iterator itqn = --QueryNodes.end();
            itqn->thisQueryNodeOnList = itqn;
          }
        }
        break;
      } // end of BigAID
      case 1:
      { // model new AID
        // std::cout << "---> Full sign comparisons with bitset!!!" << std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          float updated_sim_thres;
          for (int it = 0; it < Ntarget; it++)
          {
            // This is like counting bits after an XNOR opperation on both binary descriptors
            updated_sim_thres = (this->k > 0 && qn.last_sim > sim_thres) ? qn.last_sim : sim_thres;
            float simi = (float) ( FullDescDim - CountItFast(QueryStats[iq], TargetStats[it], FullDescDim - updated_sim_thres) );
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          if (qn.first_sim > sim_thres)
          {
  #pragma omp critical
            {
              QueryNodes.push_back(qn);
              std::list<QueryNode>::iterator itqn = --QueryNodes.end();
              itqn->thisQueryNodeOnList = itqn;
            }
          }
        }
        break;
      } // end of AID
      case 2:
      { // model AID with K neareast neighboor
        // std::cout << "---> Full sign comparisons with bitset!!!" << std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          float updated_sim_thres;
          for (int it = 0; it < Ntarget; it++)
          {
            // This is like counting bits after an XNOR opperation on both binary descriptors
            updated_sim_thres = (this->k > 0) ? qn.last_sim : 0.0;
            float simi = (float) ( FullDescDim - CountItFast(QueryStats[iq], TargetStats[it], FullDescDim - updated_sim_thres) );
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          // if ((qn.first_sim > sim_thres) && (this->k > 1) && (qn.last_sim<qn.first_sim-200.0))
          if ( (this->k > 1) && (qn.last_sim<sim_thres*qn.first_sim) )
          {
  #pragma omp critical
            {
              qn.MostSimilar_TargetNodes.pop_back();
              QueryNodes.push_back(qn);
              std::list<QueryNode>::iterator itqn = --QueryNodes.end();
              itqn->thisQueryNodeOnList = itqn;
            }
          }
        }
        break;
      } // end of AID
      case 3:
      { // model AID
        // std::cout << "---> Full sign comparisons with bitset!!!" << std::endl;
  #pragma omp parallel for default(shared)
        for (int iq = 0; iq < Nquery; iq++)
        {
          QueryNode qn(iq);
          for (int it = 0; it < Ntarget; it++)
          {
            // This is like counting bits after an XNOR opperation on both binary descriptors
            int concor = FullDescDim - (QueryStats[iq].AID ^ TargetStats[it].AID).count();
            float updated_sim_thres = (this->k > 0 && qn.last_sim > sim_thres) ? qn.last_sim : sim_thres;
            float simi = (float)concor;
            if (simi > updated_sim_thres)
              qn.Add_TargetNode(it, simi, this->k);
          }
          if (qn.first_sim > sim_thres)
          {
  #pragma omp critical
            {
              QueryNodes.push_back(qn);
              std::list<QueryNode>::iterator itqn = --QueryNodes.end();
              itqn->thisQueryNodeOnList = itqn;
            }
          }
        }
        break;
      } // end of AID
    } // end of the switch
    delete[] QueryStats;
    delete[] TargetStats;
  }

private:
  MatchersClass() : k(0), sim_thres(0.0), Nvec(0){};
};

float max_euclidean_dist(const Match & m1, const Match & m2)
{
  float left_dist = std::sqrt( std::pow(m1.x1-m2.x1,2.0) + std::pow(m1.y1-m2.y1,2.0) );
  float right_dist = std::sqrt( std::pow(m1.x2-m2.x2,2.0) + std::pow(m1.y2-m2.y2,2.0) );
  if (left_dist>right_dist)
    return left_dist;
  else
    return right_dist;
}

std::vector<Match> UniqueFilter(const std::vector<Match>& matches)
{
  std::vector<Match> uniqueM;
  bool *duplicatedM = new bool[matches.size()];
  float best_sim;
  int bestidx;
  for (int i =0; i<matches.size(); i++)
    duplicatedM[i] = false;
  for (int i =0; i<matches.size(); i++)
  {
    if (duplicatedM[i])
      continue;
    best_sim = matches[i].similarity;
    bestidx = i;
    for(int j=i+1; j<matches.size();j++)
    {
      // std::cout<< max_euclidean_dist(matches[i],matches[j])<<std::endl;
      if ( !duplicatedM[j] && max_euclidean_dist(matches[i],matches[j])<SameKPThres )
      {
        duplicatedM[j] = true;
        if (best_sim<matches[j].similarity)
        {
          bestidx = j;
          best_sim = matches[j].similarity;
        }
      }
    }
    uniqueM.push_back(matches[bestidx]);
  }
  delete[] duplicatedM;
  return uniqueM;
}

void ORSA_Filter(std::vector<Match>& matches, bool* MatchMask, float* T, int w1,int h1,int w2,int h2, bool Fundamental, const double & precision, bool verb)
{
  libNumerics::matrix<double> H(3,3);
  std::vector<int> vec_inliers;
  double nfa;
  const float nfa_max = -2;
  const int ITER_ORSA=10000;
  if (Fundamental)
    orsa::orsa_fundamental(matches, w1,h1,w2,h2, precision, ITER_ORSA,H, vec_inliers,nfa,verb);
  else
     orsa::ORSA_homography(matches, w1,h1,w2,h2, precision, ITER_ORSA,H, vec_inliers,nfa,verb);

  for (int cc = 0; cc < matches.size(); cc++ )
    MatchMask[cc] = false;

  if ( nfa < nfa_max )
  {
    for (int vi = 0; vi < vec_inliers.size(); vi++ )
      MatchMask[vec_inliers[vi]] = true;
    H /= H(2,2);

    int t = 0;
    for(int i = 0; i < H.nrow(); ++i)
        for (int j = 0; j < H.ncol(); ++j)
            T[t++] = H(i,j);
    if (verb)
    {
        printf("The two images match! %d matchings are identified. log(nfa)=%.2f.\n", (int) vec_inliers.size(), nfa);
        if (Fundamental)
          std::cout << "*************** Fundamental **************"<< std::endl;
        else
          std::cout << "*************** Homography ***************"<< std::endl;
        std::cout << H <<std::endl;
        std::cout << "******************************************"<< std::endl;
    }
  }
  else
  {
    if (verb)
        printf("The two images do not match. The matching is not significant:  log(nfa)=%.2f.\n", nfa);
  }
}



void USAC_Filter(std::vector<Match>& matches, bool* MatchMask, float* T, bool doFundamental, double precision, bool verb)
{
#ifdef _USAC
    // store the coordinates of the matching points
    std::vector<double> point_data;
    std::vector<unsigned int> prosac_data;
    prosac_data.resize(matches.size());
    point_data.resize(6*matches.size());
    for (int i = 0; i < (int) matches.size(); i++ )
    {
        point_data[6*i] = matches[i].x1;
        point_data[6*i+1] = matches[i].y1;
        point_data[6*i+3] = matches[i].x2;
        point_data[6*i+4] = matches[i].y2;
        point_data[6*i+2] = 1.0;
        point_data[6*i+5] = 1.0;
    }

    for (int cc = 0; cc < matches.size(); cc++ )
      MatchMask[cc] = false;
    libNumerics::matrix<double> F(3,3);


    char buff[4000];
    std::string cfg_file_path, dir(getcwd(buff,4000));
    if(doFundamental)
        cfg_file_path =dir+"/cpp_libs/libUSAC/fundamental.cfg";
    else
        cfg_file_path =dir+"/cpp_libs/libUSAC/homography.cfg";

    if (verb)
      std::cout<<"Usac config file: "<<cfg_file_path<<std::endl;

    if (doFundamental)
    {
        // ------------------------------------------------------------------------
        // initialize the fundamental matrix estimation problem
        ConfigParamsFund cfg;
        if ( !cfg.initParamsFromConfigFile((cfg_file_path)) )
        {
            std::cerr << "Error during initialization" << std::endl;
        }
        cfg.common.numDataPoints = matches.size();
        cfg.common.inlierThreshold = precision;
        FundMatrixEstimator* fund = new FundMatrixEstimator;
        fund->initParamsUSAC(cfg);

        // set up the fundamental matrix estimation problem

        fund->initDataUSAC(cfg);
        fund->initProblem(cfg, &point_data[0]);
        fund->solve(verb);

        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                F(i,j) = fund->final_model_params_[3*i+j];
            }
        }

        for (unsigned int i = 0; i < matches.size(); ++i)
        {
            if(fund->usac_results_.inlier_flags_[i])
                MatchMask[i] = true;
        }

        // clean up
        point_data.clear();
        prosac_data.clear();
        fund->cleanupProblem();
        delete fund;

    } 
    else
    {
        // ------------------------------------------------------------------------
        // initialize the homography estimation problem
        ConfigParamsHomog cfg;
        if ( !cfg.initParamsFromConfigFile((cfg_file_path)) )
            std::cerr << "Error during initialization" << std::endl;

        HomogEstimator* homog = new HomogEstimator;
        cfg.common.numDataPoints = matches.size();
        cfg.common.inlierThreshold = precision;
        homog->initParamsUSAC(cfg);

        // set up the homography estimation problem
        homog->initDataUSAC(cfg);
        homog->initProblem(cfg, &point_data[0]);
        homog->solve(verb);

        // write out results
        for(unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                F(i,j) = homog->final_model_params_[3*i+j];
            }
        }

        for (unsigned int i = 0; i < matches.size(); ++i)
        {
            if (homog->usac_results_.inlier_flags_[i])
                MatchMask[i] = true;
        }


        // clean up
        point_data.clear();
        prosac_data.clear();
        homog->cleanupProblem();
        delete homog;
    }
    int t = 0;
    for(int i = 0; i < F.nrow(); ++i)
        for (int j = 0; j < F.ncol(); ++j)
            T[t++] = F(i,j);

    if (verb)
    {
        if (doFundamental)
          std::cout << "*************** Fundamental **************"<< std::endl;
        else
          std::cout << "*************** Homography ***************"<< std::endl;
        std::cout << F <<std::endl;
        std::cout << "******************************************"<< std::endl;
    }

#else
    for (int cc = 0; cc < matches.size(); cc++ )
      MatchMask[cc] = false;
    std::cerr<<"The CMakeLists.txt file has the option USAC set to OFF. Please set it to ON to use USAC."<<std::endl;
#endif
}


static void
flip_descriptor (float *dst, float const *src)
{
  int const BO = 8 ;  /* number of orientation bins */
  int const BP = 4 ;  /* number of spatial bins     */
  int i, j, t ;

  for (j = 0 ; j < BP ; ++j) {
    int jp = BP - 1 - j ;
    for (i = 0 ; i < BP ; ++i) {
      int o  = BO * i + BP*BO * j  ;
      int op = BO * i + BP*BO * jp ;
      dst [op] = src[o] ;
      for (t = 1 ; t < BO ; ++t)
        dst [BO - t + op] = src [t + o] ;
    }
  }
}

  #include "cpp_libs/vl/covdet.h"
  #include "cpp_libs/vl/sift.h"


// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
  int get_vldescDim(MatchersClass* M)
  {
    return M->vl_desc_dim;
  }

  void get_vldescriptors(MatchersClass* M, float* arr)
  {
    for (int i=0; i<M->N_vlfeatData*M->vl_desc_dim;i++)
      arr[i] = M->vl_desc[i];
  }

  int get_PatchWidth(MatchersClass* M)
  {
    return M->vl_patchWidth;
  }

  void get_vlfeatPatches(MatchersClass* M, unsigned char* arr)
  {
    for (int i=0; i<M->vl_patchWidth*M->vl_patchWidth*M->N_vlfeatData;i++)
      arr[i] = M->vlfeatPatches[i];
  }

  int get_NvlfeatData(MatchersClass* M)
  {
    return M->N_vlfeatData;
  }

  void get_vlfeatData(MatchersClass* M, float* arr)
  {
    for (int i=0; i<6*M->N_vlfeatData;i++)
      arr[i] = M->vlfeatData[i];
  }

  int vlfeat(MatchersClass* M, const float *image, int numRows, int numCols, int method)
  {
    if (M->N_vlfeatData>0)
    {
      delete[] M->vlfeatData;
      delete[] M->vlfeatPatches;
      if (M->vl_desc_dim>0)
        delete[] M->vl_desc;
      M->vl_desc_dim=0;
      M->N_vlfeatData=0;
      M->vl_patchWidth=0;
    }

    vl_index patchResolution = -1 ;
    double patchRelativeExtent = -1 ;
    double patchRelativeSmoothing = -1 ;
    int descdim = -1;
    switch (method)
    {
      case 1://VL_COVDET_DESC_PATCH :
        patchResolution = 20 ;
        patchRelativeExtent = 6 ;
        patchRelativeSmoothing = 1 ;
        break ;
      case 0: //VL_COVDET_DESC_SIFT :
        /* the patch parameters are selected to match the SIFT descriptor geometry */
        patchResolution = 15 ;
        patchRelativeExtent = 7.5 ;
        patchRelativeSmoothing = 1 ;
        descdim = 128;
        break ;
      case 2://VL_COVDET_DESC_LIOP :
        patchResolution = 20 ;
        patchRelativeExtent = 4 ;
        patchRelativeSmoothing = 0.5 ;
        break ;
    }
    double boundaryMargin = patchRelativeExtent;

    // create a detector object
    VlCovDet * covdet = vl_covdet_new(VL_COVDET_METHOD_HESSIAN_LAPLACE);
    
    // set various parameters (optional)
    vl_covdet_set_first_octave(covdet, -1); // start by doubling the image resolution
    vl_covdet_set_octave_resolution(covdet, 3);
    // vl_covdet_set_peak_threshold(covdet, 0.01);
    // vl_covdet_set_edge_threshold(covdet, 10);
    vl_covdet_extract_affine_shape(covdet);
    
    // process the image and run the detector
    vl_covdet_put_image(covdet, image, (vl_size) numRows, (vl_size) numCols);
    vl_covdet_detect(covdet);

    // drop features on the margin
    vl_covdet_drop_features_outside(covdet, boundaryMargin);

    //Affine adaptation filter
    vl_covdet_extract_affine_shape(covdet);

    //orientation estimation
    vl_covdet_extract_orientations(covdet);
  
    // get feature frames
    vl_size numFeatures = vl_covdet_get_num_features(covdet);
    VlCovDetFeature const * feature = (VlCovDetFeature*) vl_covdet_get_features(covdet); // there might be an error here in the cast
    
    // get normalized feature appearance patches
    M->N_vlfeatData = (int)numFeatures;
    M->vlfeatData = new float[numFeatures*10];
    vl_size w = 2*patchResolution + 1 ;
    M->vl_patchWidth = (int) w;
    M->vlfeatPatches = new unsigned char[w*w*numFeatures];
    float * patch = new float[w*w];
    float * patchXY = new float[2*w*w];

    VlSiftFilt * sift = vl_sift_new(16, 16, 1, 3, 0) ;
    double patchStep = (double)patchRelativeExtent / patchResolution;
    M->vl_desc_dim = descdim;
    M->vl_desc = new float[descdim*numFeatures];
    float tempDesc[descdim];
    float * desc = new float[descdim];
    vl_sift_set_magnif(sift, 3.0);
    for (int i = 0 ; i < numFeatures ; i++) {
        // std::cout<<"[a11, a12, a21, a22, x, y] = "<<feature[i].frame.a11<<", "
        // <<feature[i].frame.a12<<", "<<feature[i].frame.a21<<", "<<feature[i].frame.a22
        // <<", "<<feature[i].frame.x<<", "<<feature[i].frame.y<<std::endl;
        M->vlfeatData[10*i] = feature[i].frame.a11;
        M->vlfeatData[10*i+1] = feature[i].frame.a12;
        M->vlfeatData[10*i+2] = feature[i].frame.a21;
        M->vlfeatData[10*i+3] = feature[i].frame.a22;
        M->vlfeatData[10*i+4] = feature[i].frame.x;
        M->vlfeatData[10*i+5] = feature[i].frame.y;
        M->vlfeatData[10*i+6] = feature[i].peakScore;
        M->vlfeatData[10*i+7] = feature[i].edgeScore;
        M->vlfeatData[10*i+8] = feature[i].orientationScore;
        M->vlfeatData[10*i+9] = feature[i].laplacianScaleScore;

        vl_covdet_extract_patch_for_frame(covdet,
                                          patch,
                                          patchResolution,
                                          patchRelativeExtent,
                                          patchRelativeSmoothing,
                                          feature[i].frame) ;
      for (int j=0;j<w*w;j++)
        M->vlfeatPatches[w*w*i + j] = (unsigned char) patch[j];
      
      if (method==0)
      {
        vl_imgradient_polar_f(patchXY, patchXY +1,
                                    2, 2 * w,
                                    patch, w, w, w);
        vl_sift_calc_raw_descriptor(sift,
                                          patchXY,
                                          tempDesc,
                                          (int)w, (int)w,
                                          (double)(w-1) / 2, (double)(w-1) / 2,
                                          (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) /
                                          patchStep,
                                          VL_PI / 2);

        flip_descriptor(desc, tempDesc);
        //Save descriptor now
        for (int j=0;j<descdim;j++)
          M->vl_desc[i*descdim + j] = desc[j];
      }
    }
    delete[] patch;
    delete[] desc;
    /* cleanup */
    vl_covdet_delete(covdet) ;
    return numFeatures; //  Transformation are specified by the matrices A and T embedded in feature frames. 
      // Note that this transformation maps pixels from the patch frame to the image frame.
  }


  int NumberOfFilteredMatches(MatchersClass* M)
  {
    return M->N_FilteredMatches;
  }

  void ArrayOfFilteredMatches(MatchersClass* M, int* arr)
  {
    for (int i=0; i<3*M->N_FilteredMatches;i++)
      arr[i] = M->FilteredIdxMatches[i];
  }

  void BypassGeoFilterWithConstraints(MatchersClass* M, int NNdepth, float threshold)
  {
    if (M->N_FilteredMatches>0)
      delete[] M->FilteredIdxMatches;
    M->N_FilteredMatches = 0;
    
    for(std::list<QueryNode>::const_iterator iq = M->QueryNodes.begin(); iq != M->QueryNodes.end(); ++iq) 
    {
      int countNN = 0;
      for(std::list<TargetNode>::const_iterator it = iq->MostSimilar_TargetNodes.begin(); it != iq->MostSimilar_TargetNodes.end(); ++it) 
        {
          countNN++;
          if ( (NNdepth<=0 || countNN<=NNdepth) && (it->sim_with_query>threshold) )
            M->N_FilteredMatches++;
        }
    }
    if (M->N_FilteredMatches>0)
    {
      M->FilteredIdxMatches = new int[3*M->N_FilteredMatches];
    
      int fcc = 0;
      for(std::list<QueryNode>::const_iterator iq = M->QueryNodes.begin(); iq != M->QueryNodes.end(); ++iq) 
      {
        int countNN = 0;
        for(std::list<TargetNode>::const_iterator it = iq->MostSimilar_TargetNodes.begin(); it != iq->MostSimilar_TargetNodes.end(); ++it) 
          {
            countNN++;
            if ( (NNdepth<=0 || countNN<=NNdepth) && (it->sim_with_query>threshold) )
            {
              M->FilteredIdxMatches[3*fcc] = iq->QueryIdx;
              M->FilteredIdxMatches[3*fcc+1] = it->TargetIdx;
              M->FilteredIdxMatches[3*fcc+2] = it->sim_with_query;
              fcc++;
            }
          }
      }
    }
    
  }

  void BypassGeoFilter(MatchersClass* M)
  {
    BypassGeoFilterWithConstraints(M,-1, -1.0);
  }

  void GeometricFilterFromNodes(MatchersClass* M, float* T, int w1,int h1,int w2,int h2, int type, float precision, bool verb)
  {
    if (M->N_FilteredMatches>0)
      delete[] M->FilteredIdxMatches;
    M->N_FilteredMatches = 0;

    std::vector<Match> matches;    
    for(std::list<QueryNode>::const_iterator iq = M->QueryNodes.begin(); iq != M->QueryNodes.end(); ++iq) 
      for(std::list<TargetNode>::const_iterator it = iq->MostSimilar_TargetNodes.begin(); it != iq->MostSimilar_TargetNodes.end(); ++it) 
    {
      Match match1;      
      match1.x1 = M->QueryKPs[iq->QueryIdx].x;
      match1.y1 = M->QueryKPs[iq->QueryIdx].y;
      match1.x2 = M->TargetKPs[it->TargetIdx].x;
      match1.y2 = M->TargetKPs[it->TargetIdx].y;
      match1.similarity = it->sim_with_query;
      match1.Qidx = iq->QueryIdx;
      match1.Tidx = it->TargetIdx;
      matches.push_back(match1);
    }
    
    if (matches.size()>0)
    {
      matches = UniqueFilter(matches);
      bool* MatchMask = new bool[matches.size()];
      switch (type)
      {
        case 0: //Homography
        {
          ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, false, (double)precision, verb);
          break;
        }
        case 1: // Fundamental
        {
          ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, true, (double)precision, verb);
          break;
        }
        case 2: //Homography
        {
          USAC_Filter(matches, MatchMask, T, false, (double)precision, verb);
          break;
        }
        case 3: // Fundamental
        {
          USAC_Filter(matches, MatchMask, T, true, (double)precision, verb);
          break;
        }
      }

      for (int cc = 0; cc < matches.size(); cc++ )
        if (MatchMask[cc])
          M->N_FilteredMatches++;
      M->FilteredIdxMatches = new int[3*M->N_FilteredMatches];
      int fcc = 0;
      for (int cc = 0; cc < matches.size(); cc++ )
        if (MatchMask[cc])
          {
            M->FilteredIdxMatches[3*fcc] = matches[cc].Qidx;
            M->FilteredIdxMatches[3*fcc+1] = matches[cc].Tidx;
            M->FilteredIdxMatches[3*fcc+2] = matches[cc].similarity;
            fcc++;
          }
      delete[] MatchMask;
    }
  }

  void GeometricFilter(float* scr_pts, float* dts_pts, bool* MatchMask, float* T, int N, int w1,int h1,int w2,int h2, int type, float precision, bool verb)
  {
    std::vector<Match> matches;
    for (int cc = 0; cc < N; cc++ )
    {
        Match match1;
        match1.x1 = scr_pts[cc*2];
        match1.y1 = scr_pts[cc*2+1];
        match1.x2 = dts_pts[cc*2];
        match1.y2 = dts_pts[cc*2+1];

        matches.push_back(match1);
    }

    switch (type)
    {
      case 0: //Homography
      {
        ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, false, (double)precision, verb);
        break;
      }
      case 1: // Fundamental
      {
        ORSA_Filter(matches, MatchMask, T, w1, h1, w2, h2, true, (double)precision, verb);
        break;
      }
      case 2: //Homography
      {
        USAC_Filter(matches, MatchMask, T, false, (double)precision, verb);
        break;
      }
      case 3: // Fundamental
      {
        USAC_Filter(matches, MatchMask, T, true, (double)precision, verb);
        break;
      }
    }
  }

  MatchersClass* newMatcher(int k, int full_desc_dim, float sim_thres)
  {
    if (full_desc_dim!=FullDescDim)
      std::cout<<"Desc dims don't match ("<<full_desc_dim<<"!="<<FullDescDim<<")"<<std::endl;
    return ( new MatchersClass(k, (float) sim_thres) );
  }

  void destroyMatcher(MatchersClass* M)
  {
    M->QueryNodes.clear();
    M->QueryKPs.clear();
    M->TargetKPs.clear();
    if (M->N_FilteredMatches>0)
      delete[] M->FilteredIdxMatches;
    if (M->N_vlfeatData>0)
    {
      delete[] M->vlfeatData;
      delete[] M->vlfeatPatches;
      if (M->vl_desc_dim>0)
        delete[] M->vl_desc;
    }
    delete M;
  }



  void KnnMatcher(MatchersClass* M, float* Query_pts, float* DescsQuery, int Nquery, float* Target_pts, float* DescsTarget, int Ntarget, int FastCode)
  {
    M->QueryNodes.clear();
    M->QueryKPs.clear();
    M->TargetKPs.clear();
    KeypointClass kp;
    for (int cc = 0; cc < Nquery; cc++ ){ 
      kp.x = Query_pts[cc*2]; kp.y = Query_pts[cc*2+1];
      M->QueryKPs.push_back( kp );
    }
    for (int cc = 0; cc < Ntarget; cc++ ){ 
      kp.x = Target_pts[cc*2]; kp.y = Target_pts[cc*2+1];
      M->TargetKPs.push_back( kp );
    }
    
    M->KnnMatcher(DescsQuery, Nquery, DescsTarget, Ntarget, FastCode);
  }

  int GetQueryNodeLength(QueryNode* qn)
  {
    return(qn->MostSimilar_TargetNodes.size());
  }

  QueryNode* LastQueryNode(MatchersClass* M)
  {
    if (M->QueryNodes.begin()!=M->QueryNodes.end())
      return(&*(--M->QueryNodes.end()));
    else
      return(0);
  }

  QueryNode* FirstQueryNode(MatchersClass* M)
  {
    if (M->QueryNodes.begin()!=M->QueryNodes.end())
      return(&*M->QueryNodes.begin());
    else
      return(0);
  }

  QueryNode* NextQueryNode(MatchersClass* M, QueryNode* qn)
  {
    if (qn!=0 && ++qn->thisQueryNodeOnList!=M->QueryNodes.end())
      return(&*(++qn->thisQueryNodeOnList));
    else
      return(0);
  }

  QueryNode* PrevQueryNode(MatchersClass* M, QueryNode* qn)
  {
    if (qn!=0 && qn->thisQueryNodeOnList!=M->QueryNodes.begin())
      return(&*(--qn->thisQueryNodeOnList));
    else
      return(0);
  }

  void GetData_from_QueryNode(QueryNode* qn, int* QueryIdx, int *TargetIdxes, float* simis)
  {
    QueryIdx[0] = qn->QueryIdx;
    int i = 0;
    for(std::list<TargetNode>::const_iterator it = qn->MostSimilar_TargetNodes.begin(); it != qn->MostSimilar_TargetNodes.end(); ++it)
    {
      TargetIdxes[i] = it->TargetIdx;
      simis[i] = it->sim_with_query;
      i++;
    }
  }


  
  bool ComparePatchesAC(float* P1, float *P2, int w_p, int h_p, int w1, int h1, int w2, int h2)
  {
    float** grad_mod1, **grad_mod2;
    float* grad_angle1 = gradient_angle(P1,w_p,h_p, grad_mod1);
    float* grad_angle2 = gradient_angle(P2,w_p,h_p, grad_mod2);
    
    float logNT = 1.5*log10(w1) + 1.5*log10(h1)
            + 1.5*log10(w2) + 1.5*log10(h2)
            + log10( log( 2.0 * max(w1,h1) ) / log(2.0) )
            + log10( log( 2.0 * max(w2,h2) ) / log(2.0) ) ;

    float logNFA = patch_comparison( grad_angle1, grad_angle2, w_p, h_p, logNT);

    free( (void *) grad_angle1 );
    free( (void *) grad_angle2 );

    if (logNFA<2.0)
      return true;
    else
      return false;
  }



  void FastMatCombi(int N, float* bP, int* i1_list, int *i2_list, float *patches1, float *patches2, int MemStepImg, int* last_i1_list, int *last_i2_list)
  {
    int MemStepBlock = 2*MemStepImg;
    #pragma omp parallel for firstprivate(MemStepImg, MemStepBlock, N)
    for (int k = 0; k<N; k++)
    {
      int i1 = i1_list[k];
      int i2 = i2_list[k];

      if (last_i1_list[k]!=i1)
        for (int i = 0; i<MemStepImg;i++)
          bP[k*MemStepBlock + 2*i] = patches1[i1*MemStepImg + i];

      if (last_i2_list[k]!=i2)
        for (int i = 0; i<MemStepImg;i++)
          bP[k*MemStepBlock + 2*i+1] = patches2[i2*MemStepImg + i];
    }
  }
}



