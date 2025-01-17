program check_adj_src

  use ascii_rw   ! ascii read and write module

  implicit none

  character(len=150) :: zfile,efile,nfile

  integer :: nptse, nptsn, nptsz, npts
  logical :: z_exist, e_exist, n_exist
  double precision :: t0e, dte, t0n, dtn, t0z, dtz, t0, dt
  integer, parameter :: NDIM = 40000  ! check ma_constants.f90
  double precision, dimension(NDIM) :: edata, ndata, zdata
  
  call getarg(1,zfile)
  call getarg(2,efile)
  call getarg(3,nfile)

  if (trim(zfile) =='' .or. trim(efile) =='' .or. trim(nfile) =='') then
    stop 'check_adj_src zfile efile nfile'
  endif

  inquire(file=trim(efile),exist=e_exist)
  inquire(file=trim(nfile),exist=n_exist)
  inquire(file=trim(zfile),exist=z_exist)

  ! initialization
  edata = 0; ndata = 0; zdata = 0

  ! at least one file (E,N,Z) should be present
  if (.not. e_exist .and. .not. n_exist .and. .not. z_exist) then
     stop 'At least one file should exist: zfile, efile, nfile'
  endif

  ! read in Z file
  if (z_exist) then
    call drascii(zfile,zdata,nptsz,t0z,dtz)
  endif

  ! read in E file
  if (e_exist) then
    call drascii(efile,edata,nptse,t0e,dte)
  endif

  ! read in N file
  if (n_exist) then
    call drascii(nfile,ndata,nptsn,t0n,dtn)
  endif

  ! check consistency of t0,dt,npts
  if (z_exist .and. e_exist) then
    if (abs(t0z-t0e)>1.0e-2 .or. abs(dtz-dte)>1.0e-2 .or. nptsz /= nptse) &
               stop 'check t0 and npts'
  endif
  if (e_exist .and. n_exist) then
    if (abs(t0e-t0n)>1.0e-2 .or. abs(dte-dtn)>1.0e-2 .or. nptse /= nptsn) &
               stop 'check t0 and npts'
  endif
  if (n_exist .and. z_exist) then
    if (abs(t0n-t0z)>1.0e-2 .or. abs(dtn-dtz)>1.0e-2 .or. nptsn /= nptsz) &
               stop 'check t0 and npts'
  endif

  if (e_exist) then
    t0 =t0e; dt = dte; npts = nptse
  else if (n_exist) then
    t0 =t0n; dt = dtn; npts = nptsn
  else if (z_exist) then
    t0 =t0z; dt = dtz; npts = nptsz
  endif
  if (NDIM < npts) stop 'Increase NDIM'

  ! write Z file if did not exist
  if (.not. z_exist) then
    call dwascii(zfile,zdata,npts,t0,dt)
  endif

  ! write E file if did not exist
  if (.not. e_exist) then
    call dwascii(efile,edata,npts,t0,dt)
  endif

  ! write N file if did not exist
  if (.not. n_exist) then
    call dwascii(nfile,ndata,npts,t0,dt)
  endif

end program check_adj_src
