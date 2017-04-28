using AMD
using MatrixDepot
using MathProgBase
using Clp
using PyPlot

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

  A = Problem.A
  m = size(A,1)
  b = Problem.b
  c = Problem.c
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


function get_cholesky(M)
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

  return L

end

function predictor_corrector(A, c, b, x_0, s_0,lambda_0,m_original,n_original,Problem, tol, max_iter)
    # Get system parameters
    n = size(A,2)
    m = size(A,1)

    # initialize variables
    k = 0
    x_k = copy(x_0)
    s_k = copy(s_0)
    lambda_k = copy(lambda_0)

    normalized_residual_den = norm([b;c])
    convergence = false

    iter_hist = []
    func_val_hist = []
    mu_hist = []
    norm_res_hist = []

    while k <= max_iter
        dxs = x_k./s_k
        ss = 1./s_k
        rc_k = A'*lambda_k + s_k - c
        rb_k = A*x_k - b
        rxs_k = x_k.*s_k

        # Current Duality measure
        mu = (x_k.*s_k/n)[1]

        normalized_residual = norm([rc_k;rb_k;rxs_k])/normalized_residual_den

        @show(mu, normalized_residual)
        if (mu <= tol && normalized_residual <= tol)
          convergence = false
          break
        end

        # step length for affine step
        L, ordering = get_cholesky_factor_v2(A, sparse(diagm(vec(dxs))))
        delta_lambda_aff, delta_s_aff, delta_x_aff = solve_linear_systems(L,A,dxs,ss,rc_k, rb_k, rxs_k, ordering)
        alpha_aff_prim,alpha_aff_dual = get_alpha_affine(x_k,s_k,delta_x_aff, delta_s_aff)

        # Duality measure for affine step
        mu_aff = (((x_k + alpha_aff_prim*delta_x_aff).*(s_k + alpha_aff_dual*delta_s_aff))/n)[1]

        # Set the Centering Parameter

        sigma = (mu_aff/mu)^3
        rxs_k = rxs_k + delta_x_aff.*delta_s_aff - sigma*mu

        delta_lambda, delta_s, delta_x = solve_linear_systems(L,A,dxs, ss, rc_k, rb_k, rxs_k, ordering)

        alpha_max_prim,alpha_max_dual = get_alpha_max(x_k,s_k,delta_x, delta_s)

        eta = max(0.995,1 - mu)
        alpha_primal_k = min(1,eta*alpha_max_prim)
        alpha_dual_k = min(1,eta*alpha_max_dual)

        x_k = x_k + alpha_primal_k*delta_x
        s_k = s_k + alpha_dual_k*delta_s
        lambda_k = lambda_k + alpha_dual_k*delta_lambda

        println("Iteration: \n", k)
        println((c[1:n_original])'*(Problem.lo + x_k[1:n_original]),"\n")
        iter_hist = [iter_hist;k]
        func_val_hist = [func_val_hist;(c[1:n_original])'*(Problem.lo + x_k[1:n_original])]
        mu_hist = [mu_hist;mu]
        norm_res_hist = [norm_res_hist; normalized_residual]
        k += 1
    end

    plot_graph(iter_hist, func_val_hist, mu_hist, norm_res_hist)

    #@show(eltype(x_k[1:n_original] + Problem.lo), size(convergence), size(c), size(A), size(b),size(x_k), size(lambda_k), size(s_k))

    return IplpSolution(
      vec(x_k[1:n_original] + Problem.lo),
      Bool(convergence),
      vec(c),
      sparse(A),
      vec(b),
      vec(x_k),
      vec(lambda_k),
      vec(s_k)
    )
    #return x_k, s_k, lambda_k
end

function plot_graph(iter_hist, func_val_hist, mu_hist, norm_res_hist)
  suptitle("LPnetlib/lp_afiro")
  subplot(311)
  #suptitle("Iteration vs function value")
  xlabel("Iterations")
  ylabel("function value")
  plot(iter_hist, func_val_hist)

  subplot(312)
  #suptitle("Iteration vs mu value")
  xlabel("Iterations")
  ylabel("mu value")
  plot(iter_hist,mu_hist)

  subplot(313)
  #suptitle("Iteration vs normalised residual value")
  xlabel("Iterations")
  ylabel("norm residual value")
  plot(iter_hist, norm_res_hist)

end

function solve_linear_systems(L,A,dxs,ss, rc, rb, rxs, ordering)

  lam_rhs = -rb - A*(dxs.*rc - ss.*rxs)

  delta_lambda = lam_rhs
  delta_lambda[ordering] = L'\(L\lam_rhs[ordering])

  delta_s = -rc - A' * delta_lambda
  delta_x = -ss.*rxs - dxs.*delta_s

  return delta_lambda, delta_s, delta_x

end

function get_starting_point(A, b, c)

  #x_hat, lambda_hat, s_hat arre solutions of :
  # min 0.5*x'x subject to Ax=b
  # min 0.5*s's subject to A'lambda + s = c
  L = get_cholesky(A*A')
  AA = L*L'
  d = AA\b
  x_hat = A'*d
  lambda_hat = AA\(A*c)
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

  return x_0, lambda_0, s_0
end

# This is the public interface for the problem
function iplp(Problem, tol; maxit=100000)
  m_original = size(Problem.A,1)
  n_original = size(Problem.A,2)

  sol_original = linprog(Problem.c,Problem.A,'=',Problem.b,Problem.lo,Problem.hi,ClpSolver())
  @show(sol_original)

  standard_P = convert_to_standard_form(Problem)
  #@show(full(standard_P.A), standard_P.b, standard_P.c)
  x_0, lambda_0, s_0 = get_starting_point(standard_P.A,standard_P.b,standard_P.c)
  solution = predictor_corrector(standard_P.A, standard_P.c, standard_P.b,x_0, s_0,lambda_0,m_original,n_original,Problem, tol, maxit)
  return solution
end

P = convert_matrixdepot(matrixdepot("LPnetlib/lp_bnl1", :read, meta = true))
solution = iplp(P, 1.0e-6)
@show(solution)
