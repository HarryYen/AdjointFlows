
! -------------------------------------------------------------
! edit here to change T0 and T1 on some condition 
! Note, this function is called AFTER the seismogram has been 
! read but before it is filtered.
! -------------------------------------------------------------

  subroutine modify_T0_T1_on_condition
  use seismo_variables
  implicit none

  ! do nothing

  ! adjust fstart and fend accordingly
  FSTART=1./WIN_MAX_PERIOD
  FEND=1./WIN_MIN_PERIOD

  end subroutine

! -------------------------------------------------------------
! edit here to change the time dependent properties of the 
! selection criteria
! Note, this function is called AFTER the seismogram has been 
! read and filtered.
! -------------------------------------------------------------

  subroutine set_up_criteria_arrays
  use seismo_variables 
  implicit none

  integer :: i
  double precision :: time
  double precision :: R_vel, R_time
  double precision :: Q_vel, Q_time
  double precision :: Vp_vel, Vp_time
  double precision :: Vs_vel, Vs_time
  double precision :: hypo_dist
  double precision :: env_peak_ratio_min, env_peak2_ratio_max
  double precision :: env_tshift_lock, env_pad
  double precision :: env_tmin, env_tmax
  double precision :: env_peak_obs, env_peak_syn
  double precision :: env_second_obs, env_second_syn
  double precision :: env_mean_obs, env_mean_syn
  double precision :: env_sum_obs, env_sum_syn
  double precision :: env_lag
  integer :: i_env_min, i_env_max, n_env
  integer :: i_peak_obs, i_peak_syn
  logical :: env_ok


! -----------------------------------------------------------------
! This is the basic version of the subroutine - no variation with time
! -----------------------------------------------------------------
  do i = 1, npts
    time=b+(i-1)*dt
    DLNA_LIMIT(i)=DLNA_BASE
    CC_LIMIT(i)=CC_BASE
    TSHIFT_LIMIT(i)=TSHIFT_BASE
    STALTA_W_LEVEL(i)=STALTA_BASE
    S2N_LIMIT(i)=WINDOW_S2N_BASE
  enddo

  ! these values will be used for signal2noise calculations
  ! if DATA_QUALITY=.true.
  if (DATA_QUALITY) then
    noise_start=b
    noise_end=max(ph_times(1)-WIN_MIN_PERIOD,b+dt)
    signal_start=noise_end
    signal_end=b+(npts-1)*dt
  endif


! -----------------------------------------------------------------
! Start of user-dependent portion

! This is where you reset the signal_end and noise_end values
! if you want to modify them from the defaults.
! This is also where you modulate the time dependence of the selection
! criteria.  You have access to the following parameters from the 
! seismogram itself:
!
! dt, b, kstnm, knetwk, kcmpnm
! evla, evlo, stla, stlo, evdp, azimuth, backazimuth, dist_deg, dist_km
! num_phases, ph_names, ph_times
!
! Example of modulation:
!-----------------------
! To increase s2n limit after arrival of R1 try
!
! R_vel=3.2
! R_time=dist_km/R_vel
! do i = 1, npts
!   time=b+(i-1)*dt
!   if (time.gt.R_time) then
!     S2N_LIMIT(i)=2*WINDOW_S2N_BASE
!   endif
! enddo
!
 hypo_dist=sqrt(evdp**2+dist_km**2)
 ! --------------------------------
 ! Set approximate end of rayleigh wave arrival
 R_vel=2.5
 R_time=dist_km/R_vel
 ! --------------------------------
 ! Set approximate start of love wave arrival
 Q_vel=4.2
 Q_time=dist_km/Q_vel
 ! --------------------------------
 ! Set approximate P/S arrival
 Vp_vel=7.8
 Vp_time=hypo_dist/Vp_vel
 Vs_vel=4.5
 Vs_time=hypo_dist/Vs_vel
 ! --------------------------------

 ! --------------------------------
 ! Envelope-guided gating for EGF Rayleigh packet
 ! Use peak alignment (no envelope CC) and reject if envelope is ambiguous.
 env_peak_ratio_min = 4.0d0
 env_peak2_ratio_max = 0.8d0
 env_pad = 1.0d0*WIN_MAX_PERIOD
 env_tshift_lock = max(0.25d0*WIN_MIN_PERIOD, dt)

 env_tmin = max(b, R_time - env_pad)
 env_tmax = min(b + dble(npts-1)*dt, R_time + 2.0d0 * env_pad)
 i_env_min = max(1, 1 + int((env_tmin - b)/dt))
 i_env_max = min(npts, 1 + int((env_tmax - b)/dt))

 env_ok = .true.
 if (i_env_max .le. i_env_min) env_ok = .false.

 if (env_ok) then
   env_sum_obs = 0.0d0
   env_sum_syn = 0.0d0
   env_peak_obs = -1.0d0
   env_peak_syn = -1.0d0
   env_second_obs = -1.0d0
   env_second_syn = -1.0d0
   i_peak_obs = i_env_min
   i_peak_syn = i_env_min

   ! get the max, second max of the observed and synthetic envelopes
   do i = i_env_min, i_env_max
     env_sum_obs = env_sum_obs + env_obs_lp(i)
     if (env_obs_lp(i) .gt. env_peak_obs) then
       env_second_obs = env_peak_obs
       env_peak_obs = env_obs_lp(i)
       i_peak_obs = i
     elseif (env_obs_lp(i) .gt. env_second_obs) then
       env_second_obs = env_obs_lp(i)
     endif

     env_sum_syn = env_sum_syn + env_synt_lp(i)
     if (env_synt_lp(i) .gt. env_peak_syn) then
       env_second_syn = env_peak_syn
       env_peak_syn = env_synt_lp(i)
       i_peak_syn = i
     elseif (env_synt_lp(i) .gt. env_second_syn) then
       env_second_syn = env_synt_lp(i)
     endif
   enddo

   n_env = i_env_max - i_env_min + 1
   env_mean_obs = env_sum_obs / dble(n_env)
   env_mean_syn = env_sum_syn / dble(n_env)

   if (env_peak_obs .le. 0.d0 .or. env_peak_syn .le. 0.d0) env_ok = .false.
   if (env_mean_obs .le. 0.d0 .or. env_mean_syn .le. 0.d0) env_ok = .false.
   if (env_peak_obs/env_mean_obs .lt. env_peak_ratio_min) env_ok = .false.
   if (env_peak_syn/env_mean_syn .lt. env_peak_ratio_min) env_ok = .false.
   if (env_second_obs/env_peak_obs .gt. env_peak2_ratio_max) env_ok = .false.
   if (env_second_syn/env_peak_syn .gt. env_peak2_ratio_max) env_ok = .false.
 endif

 if (.not. env_ok) then
   do i = 1, npts
     CC_LIMIT(i) = 1.1d0
   enddo
   return
 endif

 env_lag = dble(i_peak_obs - i_peak_syn) * dt
 TSHIFT_REFERENCE = env_lag
 ! --------------------------------

 ! reset the signal_end time to be the end of the Rayleigh waves
 !if (DATA_QUALITY) then
 !  signal_end=R_time
 !endif

 ! --------------------------------
 ! modulate criteria in time
 do i = 1, npts
   time=b+(i-1)*dt
   ! --------------------------------
   ! if we are beyond the Rayleigh wave, then make the all criteria stronger
   ! ratio criterion stronger
!   if (time.gt.R_time) then
!     S2N_LIMIT(i)=10*WINDOW_S2N_BASE    ! only pick big signals
!     CC_LIMIT(i)= 0.95                  ! only pick very similar signals
!     TSHIFT_LIMIT(i)=TSHIFT_BASE/3.0    ! only pick small timeshifts
!     DLNA_LIMIT(i)=DLNA_BASE/3.0        ! only pick small amplitude anomalies
!     STALTA_W_LEVEL(i)=STALTA_BASE*2.0     ! pick only distinctive arrivals
!   endif
   ! --------------------------------
   ! if we are in the surface wave times, then make the cross-correlation
   ! criterion less severe
   !if (time.gt.Q_time .and. time.lt.R_time) then
   !  CC_LIMIT(i)=0.9*CC_LIMIT(i)
   !endif

   if (time.lt.(Vp_time-10)) then
     STALTA_W_LEVEL(i)=STALTA_BASE*2.5
   endif
   if (time.gt.(Vs_time+30)) then
     STALTA_W_LEVEL(i)=STALTA_BASE*1.1
     !STALTA_W_LEVEL(i)=STALTA_BASE*3.0
     CC_LIMIT(i)= 0.85

   endif
   ! --------------------------------
   ! modulate criteria according to event depth
   !
   ! if an intermediate depth event
   if (evdp.ge.70 .and. evdp.lt.300) then
     TSHIFT_LIMIT(i)=TSHIFT_BASE*1.1
   ! if a deep event
   elseif (evdp.ge.300) then
     TSHIFT_LIMIT(i)=TSHIFT_BASE*1.7
   endif

   if (time.ge.env_tmin .and. time.le.env_tmax) then
     if (TSHIFT_LIMIT(i) .gt. env_tshift_lock) TSHIFT_LIMIT(i) = env_tshift_lock
   endif
 enddo




!
! End of user-dependent portion
! -----------------------------------------------------------------

  end subroutine
  ! -------------------------------------------------------------
