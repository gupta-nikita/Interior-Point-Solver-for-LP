using AMD
using MatrixDepot
using MathProgBase
using Clp
include("modCholesky.jl")

# to detect unbounded variables
INFINITY = 1.0e308
machine_eps = 1.1102230246251565e-16

type IplpSolution
  x::Vector{Float64} # the solution vector
  flag::Bool         # a true/false flag indicating convergence or not
  cs::Vector{Float64} # the objective vector in standard form
  As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
  bs::Vector{Float64} # the right hand side (b) in standard form
  xs::Vector{Float64} # the solution in standard form
  lam::Vector{Float64} # the solution lambda in standard form
  s::Vector{Float64} # the solution s in standard form
end


type IplpProblem
  c::Vector{Float64}
  A::SparseMatrixCSC{Float64}
  b::Vector{Float64}
  lo::Vector{Float64}
  hi::Vector{Float64}
end



type IplpProblemStandardForm
  c::Vector{Float64}
  A::SparseMatrixCSC{Float64}
  b::Vector{Float64}
end


# This gives the LP problem
function convert_matrixdepot(mmmeta::Dict{AbstractString,Any})
  key_base = sort(collect(keys(mmmeta)))[1]
  return IplpProblem(
    vec(mmmeta[key_base*"_c"]),
    mmmeta[key_base],
    vec(mmmeta[key_base*"_b"]),
    vec(mmmeta[key_base*"_lo"]),
    vec(mmmeta[key_base*"_hi"]))
end

#Convert Problem into Standard Form

function convert_to_standard_form(Problem)
  n = length(Problem.c)
  c_dash = [Problem.c;-1*Problem.c ; zeros(2*n,1)]

  A = Problem.A
  m = size(A,1)
  b = Problem.b
  hi = Problem.hi
  lo = Problem.lo

  A_dash = [A -1*A zeros(m,n) zeros(m,n);
            eye(n) -1*eye(n) -1*eye(n) zeros(n,n);
            eye(n) -1*eye(n) zeros(n,n) eye(n)]
  b_dash = [b ; lo ; hi ]

  return IplpProblemStandardForm(
          vec(c_dash),
          sparse(A_dash),
          b_dash)
end

function convert_to_standard_form_v2(Problem)
  n = length(Problem.c)

  c_dash = vec([Problem.c; vec(zeros(n,1))])

  A = Problem.A
  m = size(A,1)
  b = Problem.b
  hi = Problem.hi
  lo = Problem.lo

  A_dash = [A zeros(m,n);
            eye(n) eye(n)]

  b_dash = vec([b - A*lo; hi - lo])

  return IplpProblemStandardForm(
          c_dash,
          sparse(A_dash),
          b_dash)
end

function convert_to_standard_form_v3(Problem)
  n = length(Problem.c)

  A = Problem.A
  m = size(A,1)
  b = Problem.b
  hi = Problem.hi
  lo = Problem.lo

  As = sparse(A)
  bs = b
  cs = Problem.c
  if length(find(lo)) != 0
      b = b - A*lo
      hi = hi - lo
      bs = b
      As = sparse(A)
  end

  if length(find(hi .!= INFINITY)) != 0
      Jhigh = find(hi .!= INFINITY); 
      Vhigh = hi[Jhigh];
      jh = length(Jhigh);
      B1 = zeros(m,jh);
      B2 = eye(jh);
      B3 = zeros(jh,n); 
      B3[:,Jhigh] = B2;
      As = [A B1;B3 B2];
      As = sparse(As);
      cs = vec([c; zeros(jh,1)]);
      bs = vec([b;Vhigh]);
  end

  return IplpProblemStandardForm(
          cs,
          As,
          bs)
end
# Parameters
max_iter = 300
eta = 0.99999

function get_min_for_negative_delta(v_k, delta_v)
  min_val = Inf
  n = size(v_k,1)
  for i = 1:n
    if delta_v[i] < 0
      val = -v_k[i]/delta_v[i]
      min_val = min(min_val, val)
    end
  end
  return min_val
end

function get_alpha_affine(x_k,s_k,delta_x_aff, delta_s_aff)
    n = size(x_k,1)
    alpha_aff_prim = min(1,get_min_for_negative_delta(x_k,delta_x_aff))
    alpha_aff_dual = min(1,get_min_for_negative_delta(s_k,delta_s_aff))

    return alpha_aff_prim,alpha_aff_dual
end

function get_corrector_newton_step_rhs(rc_k,rb_k,X_k,S_k,sigma,mu,delta_x_aff, delta_s_aff)
    n = size(X_k,1)
    delta_X_aff = diagm(vec(delta_x_aff))
    delta_S_aff = diagm(vec(delta_s_aff))

    return -rc_k, -rb_k, (-1*X_k*S_k*ones(n,1) - delta_X_aff*delta_S_aff*ones(n,1) + sigma*mu*ones(n,1))
end

function get_alpha_max(x_k,s_k,delta_x, delta_s)
    n = size(x_k,1)
    #l = length(corrector_p_k)
    #delta_x = corrector_p_k[1:n,:]
    #delta_s = corrector_p_k[l-n+1:l,:]
    ## TODO:  Protect againt division by zero
    alpha_max_prim = get_min_for_negative_delta(x_k,delta_x)
    alpha_max_dual = get_min_for_negative_delta(s_k,delta_s)

    return alpha_max_prim,alpha_max_dual
end

function get_alpha(alpha_max_prim,alpha_max_dual)
    return min(1,eta*alpha_max_prim),min(1,eta*alpha_max_dual)
end


function get_cholesky_factor(A, D2)
  M = A*D2*(A')
  @show isdiag(D2)
  @show issymmetric(M)
  # @show isposdef(M)

  ordering = amd(sparse(M))
  # @show(ordering)

  L = modchol(M[ordering, ordering])
  if issparse(L)
    L = full(L)
  end
  return L, ordering
end

function get_cholesky_factor_v2(A,D2)
  M = A*D2*A'
  ordering = amd(sparse(M))

  M = M[ordering,ordering]
  n = size(M,1)
  diagM = diag(M)
  
  # Get the maximum diagonal element
  gamma = maximum(abs(diagM))
  # get the maximum off diagonal element
  xi = maximum(abs(M - diagm(diagM)))
  delta = machine_eps*(maximum([gamma + xi 1]))

  beta = sqrt(maximum([gamma xi/n machine_eps]))

  D = zeros(n,1)
  L = sparse(eye(n))
  C = zeros(n,n)

  for j = 1:n
    K = 1:j-1
    C[j,j] = M[j,j]
    if !isempty(K)
      for s = 1:j-1
        C[j,j]= C[j,j]- D[s,1]*L[j,s]*L[j,s]
      end 
    end

    if j < n
      for i = j+1:n
          C[i,j] = M[i,j]
        if !isempty(K)
          for s = 1:j-1
            C[i,j] = C[i,j] - D[s,1]*L[i,s]*L[j,s]
          end 
        end
      end

      # I can calculate D[j,j] now
      theta = maximum(abs(C[j+1:n,j]))

      D[j,1] = maximum([abs(C[j,j]) (theta/beta)^2 delta])
      
      for i = j+1:n
        L[i,j] = C[i,j]/D[j,1]
      end
    else
      D[j] = maximum([abs(C[j,j]) delta]);
    end
  end

  # Convert to the standard form of Cholesky Factorization
  for j = 1:n
    L[:,j] = L[:,j]*sqrt(D[j,1])
  end

  return L,ordering

end

function predictor_corrector(A, c, b, x_0, s_0,lambda_0,m_original,n_original,Problem)
    # Get system parameters
    n = size(A,2)
    m = size(A,1)

    # initialize variables
    k = 0
    x_k = copy(x_0)
    s_k = copy(s_0)
    lambda_k = copy(lambda_0)

    while k <= max_iter
        X_k = diagm(vec(x_k))
        S_k = diagm(vec(s_k))
        D2 = S_k\X_k
        rc_k = A'*lambda_k + s_k - c
        rb_k = A*x_k - b

        # rc_k, rb_k,rxs_k = get_predcitor_newton_step_rhs(rc_k,rb_k,X_k,S_k)

        rxs_k = -1*X_k*S_k*ones(n,1)
        # step length for affine step
        L, ordering = get_cholesky_factor_v2(A, D2)
        delta_lambda_aff, delta_s_aff, delta_x_aff = solve_linear_systems(L,A, X_k, S_k, rc_k, rb_k, rxs_k, ordering)
        alpha_aff_prim,alpha_aff_dual = get_alpha_affine(x_k,s_k,delta_x_aff, delta_s_aff)

        # Current Duality measure
        mu = (x_k'*s_k/n)[1]
        
        # Duality measure for affine step
        mu_aff = (((x_k + alpha_aff_prim*delta_x_aff)'*(s_k + alpha_aff_dual*delta_s_aff))/n)[1]
        
        # Set the Centering Parameter
        
        sigma = (mu_aff/mu)^3

        # CORRECTOR STEP

        rc_k, rb_k, rxs_k = get_corrector_newton_step_rhs(rc_k,rb_k,X_k,S_k,sigma,mu,delta_x_aff, delta_s_aff)

        #get step length
        #L = get_cholesky_factor(A, X_k, S_k)
        delta_lambda, delta_s, delta_x = solve_linear_systems(L,A, X_k, S_k, rc_k, rb_k, rxs_k, ordering)

        # @show delta_lambda
        # @show delta_s
        # @show delta_x

        # max step length
        alpha_max_prim,alpha_max_dual = get_alpha_max(x_k,s_k,delta_x, delta_s)


        # Select Eta

        eta = max(0.995,1 - mu)
        # Final Step length for primal and dual variables

        alpha_primal_k = min(1,eta*alpha_max_prim)
        alpha_dual_k = min(1,eta*alpha_max_dual)
        # @show alpha_primal_k
        # @show alpha_dual_k
        # Take a step in the final search direction
        x_k = x_k + alpha_primal_k*delta_x
        s_k = s_k + alpha_dual_k*delta_s
        lambda_k = lambda_k + alpha_dual_k*delta_lambda
        
        println("Iteration: \n", k)
        
        # @show(x_k)
        # @show(s_k)
        # @show(c'*x_k)
        #@show(x_k's_k)

        println((c[1:n_original])'*(Problem.lo + x_k[1:n_original]),"\n")
        k += 1
    end

    return x_k, s_k, lambda_k
end

function solve_linear_systems(L,A, X_k, S_k, rc, rb, rxs, ordering)

  D2 = get_D2(X_k, S_k)
  #@show(size(D2), size(rb), size(rc), size(A), size(inv(S_k)), size(rxs))
  lam_rhs = -rb - A*(X_k*inv(S_k)*rc + inv(S_k)*rxs)

  delta_lambda = L'\(L\lam_rhs[ordering])

  delta_s = -rc - A' * delta_lambda
  delta_x = -1*inv(S_k)*rxs - X_k*inv(S_k)*delta_s

  return delta_lambda, delta_s, delta_x

end

function get_starting_point(A, b, c)

  #x_hat, lambda_hat, s_hat arre solutions of :
  # min 0.5*x'x subject to Ax=b
  # min 0.5*s's subject to A'lambda + s = c
  x_hat = A'*inv(full((A*A')))*b
  lambda_hat = A*A'\A*c
  s_hat = c - A'*lambda_hat

  delta_x = max(-(3.0/2.0)*minimum(x_hat), 0)
  delta_s = max(-(3.0/2.0)*minimum(s_hat), 0)

  n = size(x_hat,1)
  x_hat = x_hat + delta_x*ones(n,1)
  s_hat = s_hat + delta_s*ones(n,1)

  delta_x = (0.5 * (x_hat'*s_hat) / (ones(n,1)'*s_hat))[1]
  delta_s = (0.5 * (x_hat'*s_hat) / (ones(n,1)'*x_hat))[1]

  x_0 = x_hat + delta_x*ones(n,1)
  lambda_0 = lambda_hat
  s_0 = s_hat + delta_s*ones(n,1)

  #@show(x_0, lambda_0, s_0)
  return x_0, lambda_0, s_0
end

# This is the public interface for the problem
function iplp(Problem, tol; maxit=100)
  m_original = size(Problem.A,1)
  n_original = size(Problem.A,2)

  standard_P = convert_to_standard_form_v3(Problem)
  x_0, lambda_0, s_0 = get_starting_point(standard_P.A,standard_P.b,standard_P.c)
  x_sol, s_sol, lambda_sol = predictor_corrector(standard_P.A, standard_P.c, standard_P.b,x_0, s_0,lambda_0,m_original,n_original,Problem)
end