! =====================================================================
! ===========This code is not from the original source code!===========
! =====================================================================
! Purpose:
! In the original source code (get_direction_sd), 
! they directly calculate the direction in that subroutine (d = -g).
! But in our flow, we have already calculated the direction using
! either LBFGS or SD. So, we just need to get the direction without
! calculating it again.
! --------------------------------------------------------------------
! This subroutine will be called in model_update.f90
! Lastest updating: 2024/08/28.
! Modified by Hung-Yu Yen, IES, AS.
! =====================================================================


subroutine get_direction_without_calculation_sd()

    
    use tomography_kernels_iso
  
    implicit none
    ! local parameters
    real(kind=CUSTOM_REAL):: r,max,depth_max
    ! gradient vector norm ( v^T * v )
    real(kind=CUSTOM_REAL) :: norm_bulk,norm_beta,norm_rho
    real(kind=CUSTOM_REAL) :: norm_bulk_sum,norm_beta_sum,norm_rho_sum
    real(kind=CUSTOM_REAL) :: minmax(4)
    real(kind=CUSTOM_REAL) :: min_beta,min_rho,max_beta,max_rho,min_bulk,max_bulk
    integer :: iglob
    integer :: i,j,k,ispec,ier
  
    ! allocate arrays for storing gradient
    ! isotropic arrays
    allocate(model_dbulk(NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
    if (ier /= 0) call exit_MPI_without_rank('error allocating array 1046')
    allocate(model_dbeta(NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
    if (ier /= 0) call exit_MPI_without_rank('error allocating array 1047')
    allocate(model_drho(NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
    if (ier /= 0) call exit_MPI_without_rank('error allocating array 1048')
    if (ier /= 0) stop 'error allocating gradient arrays'
  
    ! initializes arrays
    model_dbulk = 0.0_CUSTOM_REAL
    model_dbeta = 0.0_CUSTOM_REAL
    model_drho = 0.0_CUSTOM_REAL
  
    ! initializes kernel maximum
    max = 0._CUSTOM_REAL
  
    ! gradient in negative direction for steepest descent
    do ispec = 1, NSPEC
      do k = 1, NGLLZ
        do j = 1, NGLLY
          do i = 1, NGLLX
  
              ! for bulk
              model_dbulk(i,j,k,ispec) = kernel_bulk(i,j,k,ispec)
  
              ! for shear
              model_dbeta(i,j,k,ispec) = kernel_beta(i,j,k,ispec)
  
              ! for rho
              model_drho(i,j,k,ispec) = kernel_rho(i,j,k,ispec)
  
              ! determines maximum kernel beta value within given radius
              if (USE_DEPTH_RANGE_MAXIMUM) then
                ! get depth of point (assuming z in vertical direction, up in positive direction)
                iglob = ibool(i,j,k,ispec)
                r = z(iglob)
  
                ! stores maximum kernel betav value in this depth slice, since betav is most likely dominating
                if (r < R_TOP .and. r > R_BOTTOM) then
                  ! shear kernel value
                  max_beta = abs( kernel_beta(i,j,k,ispec) )
                  if (max < max_beta) then
                    max = max_beta
                    depth_max = r
                  endif
                endif
              endif
  
          enddo
        enddo
      enddo
    enddo
  
    ! statistics
    call min_all_cr(minval(model_dbulk),min_bulk)
    call max_all_cr(maxval(model_dbulk),max_bulk)
  
    call min_all_cr(minval(model_dbeta),min_beta)
    call max_all_cr(maxval(model_dbeta),max_beta)
  
    call min_all_cr(minval(model_drho),min_rho)
    call max_all_cr(maxval(model_drho),max_rho)
  
    if (myrank == 0) then
      print *,'initial gradient:'
      print *,'  a min/max   : ',min_bulk,max_bulk
      print *,'  beta min/max: ',min_beta,max_beta
      print *,'  rho min/max : ',min_rho,max_rho
      print *
    endif
  
    ! statistics output
    if (PRINT_STATISTICS_FILES .and. myrank == 0) then
      open(IOUT,file=trim(OUTPUT_STATISTICS_DIR)//'statistics_gradient_minmax',status='unknown')
      write(IOUT,*) '#min_beta #max_beta #min_bulk #max_bulk #min_rho #max_rho'
      write(IOUT,'(6e24.12)') min_beta, max_beta, min_bulk, max_bulk, min_rho, max_rho
      close(IOUT)
    endif
  
    ! determines maximum kernel betav value within given radius
    if (USE_DEPTH_RANGE_MAXIMUM) then
      ! maximum of all processes stored in max_vsv
      call max_all_cr(max,max_beta)
      max = max_beta
    endif
  
    ! determines step length based on maximum gradient value (either shear or bulk)
    if (myrank == 0) then
  
      ! determines maximum kernel betav value within given radius
      if (USE_DEPTH_RANGE_MAXIMUM) then
        print *,'  using depth maximum: '
        print *,'  between depths (top/bottom)   : ',R_TOP,R_BOTTOM
        print *,'  maximum kernel value          : ',max
        print *,'  depth of maximum kernel value : ',depth_max
        print *
      else
        ! maximum gradient values
        minmax(1) = abs(min_beta)
        minmax(2) = abs(max_beta)
        minmax(3) = abs(min_bulk)
        minmax(4) = abs(max_bulk)
  
        ! maximum value of all kernel maxima
        max = maxval(minmax)
      endif
      print *,'step length:'
      print *,'  using kernel maximum: ',max
  
      ! checks maximum value
      if (max < 1.e-25) stop 'Error maximum kernel value too small for update'
  
      ! chooses step length such that it becomes the desired, given step factor as inputted
      step_length = step_fac/max
  
      print *,'  step length value   : ',step_length
      print *
    endif
    call bcast_all_singlecr(step_length)
  
  
    ! gradient length sqrt( v^T * v )
    norm_bulk = sum( model_dbulk * model_dbulk )
    norm_beta = sum( model_dbeta * model_dbeta )
    norm_rho = sum( model_drho * model_drho )
  
    call sum_all_cr(norm_bulk,norm_bulk_sum)
    call sum_all_cr(norm_beta,norm_beta_sum)
    call sum_all_cr(norm_rho,norm_rho_sum)
  
    if (myrank == 0) then
      norm_bulk = sqrt(norm_bulk_sum)
      norm_beta = sqrt(norm_beta_sum)
      norm_rho = sqrt(norm_rho_sum)
  
      print *,'norm model updates:'
      print *,'  a   : ',norm_bulk
      print *,'  beta: ',norm_beta
      print *,'  rho : ',norm_rho
      print *
    endif
  
    ! statistics output
    if (PRINT_STATISTICS_FILES .and. myrank == 0) then
      open(IOUT,file=trim(OUTPUT_STATISTICS_DIR)//'statistics_vs_vp_rho_sum',status='unknown')
      write(IOUT,*) '#norm_beta #norm_bulk #norm_rho'
      write(IOUT,'(3e24.12)') norm_beta, norm_bulk, norm_rho
      close(IOUT)
    endif
  
    ! multiply model updates by a subjective factor that will change the step
    model_dbulk(:,:,:,:) = step_length * model_dbulk(:,:,:,:)
    model_dbeta(:,:,:,:) = step_length * model_dbeta(:,:,:,:)
    model_drho(:,:,:,:) = step_length * model_drho(:,:,:,:)
  
    ! statistics
    call min_all_cr(minval(model_dbulk),min_bulk)
    call max_all_cr(maxval(model_dbulk),max_bulk)
  
    call min_all_cr(minval(model_dbeta),min_beta)
    call max_all_cr(maxval(model_dbeta),max_beta)
  
    call min_all_cr(minval(model_drho),min_rho)
    call max_all_cr(maxval(model_drho),max_rho)
  
    if (myrank == 0) then
      print *,'scaled gradient:'
      print *,'  a min/max   : ',min_bulk,max_bulk
      print *,'  beta min/max: ',min_beta,max_beta
      print *,'  rho min/max : ',min_rho,max_rho
      print *
    endif
    call synchronize_all()
  
    ! statistics output
    if (PRINT_STATISTICS_FILES .and. myrank == 0) then
      open(IOUT,file=trim(OUTPUT_STATISTICS_DIR)//'statistics_scaled_gradient',status='unknown')
      write(IOUT,*) '#min_beta #max_beta #min_bulk #max_bulk #min_rho #max_rho'
      write(IOUT,'(6e24.12)') min_beta,max_beta,min_bulk,max_bulk,min_rho,max_rho
      close(IOUT)
    endif
    
  
    
    end subroutine get_direction_without_calculation_sd



    subroutine get_direction_without_calculation_lbfgs()

    
        use tomography_kernels_iso
      
        implicit none
        ! local parameters
        real(kind=CUSTOM_REAL):: r,max,depth_max
        ! gradient vector norm ( v^T * v )
        real(kind=CUSTOM_REAL) :: norm_bulk,norm_beta,norm_rho
        real(kind=CUSTOM_REAL) :: norm_bulk_sum,norm_beta_sum,norm_rho_sum
        real(kind=CUSTOM_REAL) :: minmax(4)
        real(kind=CUSTOM_REAL) :: min_beta,min_rho,max_beta,max_rho,min_bulk,max_bulk
        integer :: iglob
        integer :: i,j,k,ispec,ier
      
        ! allocate arrays for storing gradient
        ! isotropic arrays
        allocate(model_dbulk(NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
        if (ier /= 0) call exit_MPI_without_rank('error allocating array 1046')
        allocate(model_dbeta(NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
        if (ier /= 0) call exit_MPI_without_rank('error allocating array 1047')
        allocate(model_drho(NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
        if (ier /= 0) call exit_MPI_without_rank('error allocating array 1048')
        if (ier /= 0) stop 'error allocating gradient arrays'
      
        ! initializes arrays
        model_dbulk = 0.0_CUSTOM_REAL
        model_dbeta = 0.0_CUSTOM_REAL
        model_drho = 0.0_CUSTOM_REAL
      
        ! initializes kernel maximum
        max = 0._CUSTOM_REAL
      
        ! gradient in negative direction for steepest descent
        do ispec = 1, NSPEC
          do k = 1, NGLLZ
            do j = 1, NGLLY
              do i = 1, NGLLX
      
                  ! for bulk
                  model_dbulk(i,j,k,ispec) = kernel_bulk(i,j,k,ispec)
      
                  ! for shear
                  model_dbeta(i,j,k,ispec) = kernel_beta(i,j,k,ispec)
      
                  ! for rho
                  model_drho(i,j,k,ispec) = kernel_rho(i,j,k,ispec)
      
                  ! determines maximum kernel beta value within given radius
                  if (USE_DEPTH_RANGE_MAXIMUM) then
                    ! get depth of point (assuming z in vertical direction, up in positive direction)
                    iglob = ibool(i,j,k,ispec)
                    r = z(iglob)
      
                    ! stores maximum kernel betav value in this depth slice, since betav is most likely dominating
                    if (r < R_TOP .and. r > R_BOTTOM) then
                      ! shear kernel value
                      max_beta = abs( kernel_beta(i,j,k,ispec) )
                      if (max < max_beta) then
                        max = max_beta
                        depth_max = r
                      endif
                    endif
                  endif
      
              enddo
            enddo
          enddo
        enddo
      
        ! statistics
        call min_all_cr(minval(model_dbulk),min_bulk)
        call max_all_cr(maxval(model_dbulk),max_bulk)
      
        call min_all_cr(minval(model_dbeta),min_beta)
        call max_all_cr(maxval(model_dbeta),max_beta)
      
        call min_all_cr(minval(model_drho),min_rho)
        call max_all_cr(maxval(model_drho),max_rho)
      
        if (myrank == 0) then
          print *,'initial gradient:'
          print *,'  a min/max   : ',min_bulk,max_bulk
          print *,'  beta min/max: ',min_beta,max_beta
          print *,'  rho min/max : ',min_rho,max_rho
          print *
        endif
      
        ! statistics output
        if (PRINT_STATISTICS_FILES .and. myrank == 0) then
          open(IOUT,file=trim(OUTPUT_STATISTICS_DIR)//'statistics_gradient_minmax',status='unknown')
          write(IOUT,*) '#min_beta #max_beta #min_bulk #max_bulk #min_rho #max_rho'
          write(IOUT,'(6e24.12)') min_beta, max_beta, min_bulk, max_bulk, min_rho, max_rho
          close(IOUT)
        endif
      
        ! determines maximum kernel betav value within given radius
        if (USE_DEPTH_RANGE_MAXIMUM) then
          ! maximum of all processes stored in max_vsv
          call max_all_cr(max,max_beta)
          max = max_beta
        endif
      
        ! determines step length based on maximum gradient value (either shear or bulk)
        if (myrank == 0) then
      
          ! determines maximum kernel betav value within given radius
          if (USE_DEPTH_RANGE_MAXIMUM) then
            print *,'  using depth maximum: '
            print *,'  between depths (top/bottom)   : ',R_TOP,R_BOTTOM
            print *,'  maximum kernel value          : ',max
            print *,'  depth of maximum kernel value : ',depth_max
            print *
          else
            ! maximum gradient values
            minmax(1) = abs(min_beta)
            minmax(2) = abs(max_beta)
            minmax(3) = abs(min_bulk)
            minmax(4) = abs(max_bulk)
      
            ! maximum value of all kernel maxima
            max = maxval(minmax)
          endif
          print *,'step length:'
          print *,'  using kernel maximum: ',max
      
          ! checks maximum value
          if (max < 1.e-25) stop 'Error maximum kernel value too small for update'
      
          ! chooses step length
          step_length = step_fac
      
          print *,'  step length value   : ',step_length
          print *
        endif
        call bcast_all_singlecr(step_length)
      
      
        ! gradient length sqrt( v^T * v )
        norm_bulk = sum( model_dbulk * model_dbulk )
        norm_beta = sum( model_dbeta * model_dbeta )
        norm_rho = sum( model_drho * model_drho )
      
        call sum_all_cr(norm_bulk,norm_bulk_sum)
        call sum_all_cr(norm_beta,norm_beta_sum)
        call sum_all_cr(norm_rho,norm_rho_sum)
      
        if (myrank == 0) then
          norm_bulk = sqrt(norm_bulk_sum)
          norm_beta = sqrt(norm_beta_sum)
          norm_rho = sqrt(norm_rho_sum)
      
          print *,'norm model updates:'
          print *,'  a   : ',norm_bulk
          print *,'  beta: ',norm_beta
          print *,'  rho : ',norm_rho
          print *
        endif
      
        ! statistics output
        if (PRINT_STATISTICS_FILES .and. myrank == 0) then
          open(IOUT,file=trim(OUTPUT_STATISTICS_DIR)//'statistics_vs_vp_rho_sum',status='unknown')
          write(IOUT,*) '#norm_beta #norm_bulk #norm_rho'
          write(IOUT,'(3e24.12)') norm_beta, norm_bulk, norm_rho
          close(IOUT)
        endif
      
        ! multiply model updates by a subjective factor that will change the step
        model_dbulk(:,:,:,:) = step_length * model_dbulk(:,:,:,:)
        model_dbeta(:,:,:,:) = step_length * model_dbeta(:,:,:,:)
        model_drho(:,:,:,:) = step_length * model_drho(:,:,:,:)
      
        ! statistics
        call min_all_cr(minval(model_dbulk),min_bulk)
        call max_all_cr(maxval(model_dbulk),max_bulk)
      
        call min_all_cr(minval(model_dbeta),min_beta)
        call max_all_cr(maxval(model_dbeta),max_beta)
      
        call min_all_cr(minval(model_drho),min_rho)
        call max_all_cr(maxval(model_drho),max_rho)
      
        if (myrank == 0) then
          print *,'scaled gradient:'
          print *,'  a min/max   : ',min_bulk,max_bulk
          print *,'  beta min/max: ',min_beta,max_beta
          print *,'  rho min/max : ',min_rho,max_rho
          print *
        endif
        call synchronize_all()
      
        ! statistics output
        if (PRINT_STATISTICS_FILES .and. myrank == 0) then
          open(IOUT,file=trim(OUTPUT_STATISTICS_DIR)//'statistics_scaled_gradient',status='unknown')
          write(IOUT,*) '#min_beta #max_beta #min_bulk #max_bulk #min_rho #max_rho'
          write(IOUT,'(6e24.12)') min_beta,max_beta,min_bulk,max_bulk,min_rho,max_rho
          close(IOUT)
        endif
        
      
        
        end subroutine get_direction_without_calculation_lbfgs