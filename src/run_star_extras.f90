! *****************************************************
!
!   run_star_extras file for Astero Across the HRD Labs
!
! *****************************************************

      module run_star_extras

      use star_lib
      use star_def
      use const_def
      use math_lib

      use gyre_lib

      implicit none

      !! GYRE stuff
      real(dp), allocatable, save :: frequencies(:,:)
      real(dp), allocatable, save :: inertias(:)
      real(dp), allocatable, save :: etas(:)
      real(dp), allocatable, save :: freq(:)
      ! Radial displacement eigenfunctions 
      real(dp), allocatable, save :: xi_r_radial(:) !! but won't use this
      ! real(dp), allocatable, save :: xi_r_dipole(:)
      logical, save :: gyre_has_run

      integer :: stop_phase

      real(dp) :: obs_Teff_min, obs_Teff_max, obs_L_min, obs_L_max

      ! these routines are called by the standard run_star check_model
      contains

      subroutine extras_controls(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return


         s% extras_startup => extras_startup
         s% extras_start_step => extras_start_step
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns  

         s% how_many_extra_history_header_items => how_many_extra_history_header_items
         s% data_for_extra_history_header_items => data_for_extra_history_header_items
         s% how_many_extra_profile_header_items => how_many_extra_profile_header_items
         s% data_for_extra_profile_header_items => data_for_extra_profile_header_items

      end subroutine extras_controls


      subroutine extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return


         ! Initialize GYRE
         call gyre_init('gyre.in')

         ! Set constants

         call gyre_set_constant('G_GRAVITY', standard_cgrav)
         call gyre_set_constant('C_LIGHT', clight)
         call gyre_set_constant('A_RADIATION', crad)

         call gyre_set_constant('M_SUN', Msun)
         call gyre_set_constant('R_SUN', Rsun)
         call gyre_set_constant('L_SUN', Lsun)

         call gyre_set_constant('GYRE_DIR', TRIM(mesa_dir)//'/gyre/gyre')

         !! changed all from 50 (from Dora's project) to 10

         allocate(inertias(10))
         allocate(etas(10))
         allocate(freq(10))
         allocate(frequencies(2,10))

      end subroutine extras_startup



      integer function extras_start_step(id)
         integer, intent(in) :: id
         integer :: ierr
         integer :: k !, nz 
         type (star_info), pointer :: s 
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         gyre_has_run = .false.

         do k = 1, 10 !50
            frequencies(1,k) = 0   !! l = 0 ! radial 
            frequencies(2,k) = 0   !! l = 1 ! dipole
            inertias(k) = 0

            !! initialize array for etas and freq
            etas(k) = 0
            freq(k) = 0
         end do

        obs_Teff_min = s% x_ctrl(3)
        obs_Teff_max = s% x_ctrl(4) 
        obs_L_min = s% x_ctrl(5)
        obs_L_max = s% x_ctrl(6)

        if (  (s% Teff >= obs_Teff_min) .and. &
              (s% Teff <= obs_Teff_max) .and. &
              (s% L( s% nz ) >= obs_L_min) .and. & 
              (s% L( s% nz) <= obs_L_max) ) then
            
            write(*,*) 'model within the observational bounds; reducing timestep to 10 years...'
            s%max_years_for_timestep = 10
        endif


         extras_start_step = 0
      end function extras_start_step


      ! returns either keep_going, retry, or terminate.
      integer function extras_check_model(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         extras_check_model = keep_going         
         if (.false. .and. s% star_mass_h1 < 0.35d0) then
            ! stop when star hydrogen mass drops to specified level
            extras_check_model = terminate
            write(*, *) 'have reached desired hydrogen mass'
            return
         end if

         if (s%x_logical_ctrl(1)) then
             call run_gyre(id, ierr)
         endif

         if (extras_check_model == terminate) s% termination_code = t_extras_check_model

      end function extras_check_model


      integer function how_many_extra_history_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         how_many_extra_history_columns = 12 !10 !100 !! adding in outer cell gamma1
      end function how_many_extra_history_columns



      subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer :: phase_of_evolution
         integer :: k
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return


         !! GYRE stuff
         do k = 1, 10
            write (names(k),    '(A,I0)') 'nu_radial_', k-1 
        end do
       
        write( names(11),  '(A,I0)') 'gamma1_top' 
        vals(11) = s% gamma1(s% nz)
        write( names(12),  '(A,I0)') 'gamma1_bottom'
        vals(12) = s% gamma1(1)  


         !! remove dipoles; not necessary and causes crash on the RGB
         if (s%x_logical_ctrl(1)) then

          
           !! ensure that GYRE is being run at every timestep
           if (.NOT. gyre_has_run) then
              write(*,*) 'calling run_gyre'
               call run_gyre(id, ierr)
           endif

            ! save the frequencies of the radial (not dipole) modes 
            do k = 1, 10
               vals(k) = frequencies(1, k)
            end do

         else
            ! write out zeros for the 2*50 columns 
            do k = 1, 10 !100
                vals(k) = -99
            end do

         endif

      end subroutine data_for_extra_history_columns



      integer function how_many_extra_profile_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_columns = 0
      end function how_many_extra_profile_columns



      subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
         integer, intent(in) :: id, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return


      end subroutine data_for_extra_profile_columns


      integer function how_many_extra_history_header_items(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_header_items = 0
      end function how_many_extra_history_header_items


      subroutine data_for_extra_history_header_items(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         type(star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         call star_ptr(id,s,ierr)
         if(ierr/=0) return
      end subroutine data_for_extra_history_header_items


      integer function how_many_extra_profile_header_items(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_header_items = 0
      end function how_many_extra_profile_header_items


      subroutine data_for_extra_profile_header_items(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(n)
         type(star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         call star_ptr(id,s,ierr)
         if(ierr/=0) return
      end subroutine data_for_extra_profile_header_items



      integer function extras_finish_step(id)
         integer, intent(in) :: id
         integer :: ierr, phase_of_evolution

         integer :: k, best_k, stop_phase
         real(dp) :: best_freq

         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
    
         extras_finish_step = keep_going

         if (extras_finish_step == terminate) s% termination_code = t_extras_finish_step

      end function extras_finish_step


      subroutine extras_after_evolve(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine extras_after_evolve



      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !
      ! GYRE-on-the-fly
      !
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine run_gyre (id, ierr)

         integer, intent(in)  :: id
         integer, intent(out) :: ierr

         real(dp), allocatable :: global_data(:)
         real(dp), allocatable :: point_data(:,:)
         integer               :: ipar(0)
         real(dp)              :: rpar(0)

         ! Pass model data to GYRE

         call star_get_pulse_data(id, 'GYRE', .FALSE., .TRUE., .TRUE., &
              global_data, point_data, ierr)
         if (ierr /= 0) then
            print *,'Failed when calling star_get_pulse_data'
            return
         end if

         call gyre_set_model(global_data, point_data, 101)

    
         ! Run GYRE to get modes
         !!   gyre_get_modes(l, user_sub, ipar, rpar) !! normal version
         !!   gyre_get_modes(l, user_sub, ipar, rpar, mode_flag) !! mod version
         call gyre_get_modes(0, process_mode, ipar, rpar)
         
         !! non-adiabatic is never triggered, even when .false. is provided,
         !! because adiabatic is being implied somewhere ?

         !call gyre_get_modes(1, process_mode, ipar, rpar)

         gyre_has_run = .true.

      contains

         subroutine process_mode(md, ipar, rpar, retcode)
         !!! unclear whether incuding my own version of process_mode() here
         !! undoes the work of making gyre_get_modes() in gyre_lib.f90
         !! accept the adiabatic/non-adiabtic flag in the first place

            type(mode_t), intent(in) :: md
            !integer :: md
            integer, intent(inout)   :: ipar(:)
            real(dp), intent(inout)  :: rpar(:)
            integer, intent(out)     :: retcode
            integer :: k !, n


            type (star_info), pointer :: s
            ierr = 0
            call star_ptr(id, s, ierr)
            if (ierr /= 0) return

            if (md%n_p >= 0 .and. md%n_p <= 10 .and. (md%l == 0) ) then

                ! Print out degree, radial order, mode inertia, and frequency
                !! eta should be nonzero only during the instibality strip
                ! print *, '  (1)  Found mode: l, n_p, n_g, n_pg, E, nu, eta =      ', &
                !     md%l, md%n_p, md%n_g, md%n_pg, md%E_norm(), REAL(md%freq('CYC_PER_DAY')),  &
                !     md%eta()


              ! print *, md%l+1, md%n_p, md%freq('CYC_PER_DAY')      
             !frequencies(md%l+1, md%n_p) = md%freq('HZ') 
             frequencies(1, md%n_p) = md%freq('HZ')

            !------------------------------------------
            !
            ! we don't need any of this unless we want
            ! the full eigenfunction as a function of radius
            !
            !-----------------------------------------
              ! if (md%n_p == s% x_integer_ctrl(1)) then ! store the eigenfunction 
              !    if (allocated(xi_r_radial)) deallocate(xi_r_radial)
              !    allocate(xi_r_radial(md%n))
              !    do k = 1, md%n
              !       xi_r_radial(k) = md%xi_r(k)
              !    end do
              !    xi_r_radial = xi_r_radial(md%n:1:-1)
              ! end if

            end if

            retcode = 0

         end subroutine process_mode
      
      end subroutine run_gyre




      end module run_star_extras
