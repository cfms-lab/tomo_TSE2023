#pragma once
#include "STomoNV_Base.h"
#include "STomoVoxel.h"

using namespace Tomo;

class DLLEXPORT STomoNV_INT3 : public STomoNV_Base
{
public:
  STomoNV_INT3();
  STomoNV_INT3(const STomoNV_INT3& Source);
  void	operator=(const STomoNV_INT3& Source);
  void	_Copy(const STomoNV_INT3& Source);
  ~STomoNV_INT3();

  void	Reset(void);
  void	Init(void);

  STomoVoxelSpaceInfo voxel_info;

  //virtual functions of STomoNV_Base
  void  Rotate(void);
  void  Pixelize(const TVVector& CVV_vxls);
    void  rotateCVVoxel(const TVVector& CVV_vxls);
  void  Pairing(void);
  void  GenerateBed(void);
    bool  IsBedCandidate(int  X , int  Y);
  void  Calculate(void);
  TPVector   GetSSPixels(bool _bUseExplicitSS);//rendering
  TPVector  slotsToPxls(enumPixelType _type);//for rendering. time consuming. 

  void  vslotPair(const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, FLOAT32 theta_c);

  void  triVoxel(
    FLOAT32* _v0,
    FLOAT32* _v1,
    FLOAT32* _v2,
    FLOAT32* n0,
    FLOAT32* n1,
    FLOAT32* n2,
    /*output */ SLOT_BUFFER_TYPE* _Type_buffer);//for later CUDA version.


  void  createAlBePxls(         const size_t S_W, SLOT_BUFFER_TYPE& S_L,  SLOT_BUFFER_TYPE* curr_Pxl_slot);//Todo: �μ� ������ ����ü �ϳ��� ����.
    void  splitAlBe(            const size_t S_W, const size_t S_L,       SLOT_BUFFER_TYPE* curr_Pxl_slot);
    void  matchAlBeAlternation( const size_t S_W, SLOT_BUFFER_TYPE& S_L,  SLOT_BUFFER_TYPE* curr_Pxl_slot);
      inline int _xorAlBe(int _iTypeByte);
      bool  _hasPxlBetween(     const size_t S_W, SLOT_BUFFER_TYPE& S_L,  SLOT_BUFFER_TYPE* curr_Pxl_slot, int z0, int z1, int iTypeByte);

    void  matchPairNumber_Al_Be(const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);

    size_t  matchPairNumber_SS( const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, int nvb_type_byte, int be_type_byte);

    size_t countType(           const size_t S_W, const size_t S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, int typeByte = 0xff);
    SLOT_BUFFER_TYPE sumType(       const size_t S_W, const size_t S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, int typeByte = 0xff);
    void  removeZNearPxls(      const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, int typeByte);
    void  sortSlotByZ(          const size_t S_W, const SLOT_BUFFER_TYPE S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);
    SLOT_BUFFER_TYPE erasePxl(  const size_t S_W, const size_t S_L, int p_id, SLOT_BUFFER_TYPE* curr_Pxl_slot);
    void  insertPxl(            const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot,
      SLOT_BUFFER_TYPE _type, SLOT_BUFFER_TYPE _z, SLOT_BUFFER_TYPE nZ);

    void  createTCPixels(       const size_t S_W, const size_t S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);
    size_t  createShadowCastor( const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, FLOAT32 theta_c, bool _bUseExplicitSS=false);
    void  createShadowAcceptor( const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, int nvb_type_byte, int nva_type_byte);
      
    void  createVoPixels(       const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);
    void  createVss_Explicit(   const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);
    void  createVss_Implicit(   const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);
  
  void  matchAlBePairBriefly(const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot);
  void  createShadowBriefly(const size_t S_W, SLOT_BUFFER_TYPE& S_L, SLOT_BUFFER_TYPE* curr_Pxl_slot, FLOAT32 theta_c);
};

