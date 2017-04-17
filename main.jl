
using MatrixDepot

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
  c_dash = vec([Problem.c;-1*Problem.c ; vec(zeros(2*n,1))])

  A = Problem.A
  m = size(A,1)
  b = Problem.b
  hi = Problem.hi
  lo = Problem.lo
  @show size(A)
  A_dash = [A -1*A zeros(m,n) zeros(m,n); 
            eye(n) 1*eye(n) -1*eye(n) zeros(n,n); 
            eye(n) -1*eye(n) zeros(n,n) eye(n)]
  b_dash = [b ; hi ; lo ]
  return IplpProblemStandardForm(
          vec(c_dash),
          sparse(A_dash),
          b_dash)
end

# Parameters
max_iter = 10
eta = 1


function get_X(x_k)
    return diagm(vec(x_k))
end

function get_S(s_k)
    return diagm(vec(s_k))
end

function get_jacobian(X_k,S_k,A)
    n = size(X_k,1)
    m = size(A,1)
    J =  [zeros(n,n) A' eye(n);
           A zeros(m,m) zeros(m,n);
           S_k zeros(n,m) X_k ]
    return J
end

function get_rc(A,lambda_k,s_k,c)
    return A'*lambda_k + s_k - c
end

function get_rb(A,x_k,b)
    return A*x_k - b
end

function get_predcitor_newton_step_rhs(rc_k,rb_k,X_k,S_k)
    n = size(X_k,1)
    return [-rc_k ; -rb_k ; -X_k*S_k*ones(n,1)]
end

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

function get_alpha_affine(x_k,s_k,predictor_p_k)
    n = size(x_k,1)
    l = size(predictor_p_k,1)
    delta_x_aff = predictor_p_k[1:n,:]
    delta_s_aff = predictor_p_k[l-n+1:l,:]
    ## TODO:  Protect againt division by zero
    alpha_aff_prim = min(1,get_min_for_negative_delta(x_k,delta_x_aff))
    alpha_aff_dual = min(1,get_min_for_negative_delta(s_k,delta_s_aff))

    return alpha_aff_prim,alpha_aff_dual
end

function get_mu(x_k,s_k)
    n = size(x_k,1)
    return (x_k'*s_k/n)[1]
end

function get_mu_aff(alpha_aff_prim,alpha_aff_dual,x_k,s_k,predictor_p_k)
    #n = size(X_k,1)
    l = length(predictor_p_k)
    delta_x_aff = predictor_p_k[1:n]
    delta_s_aff = predictor_p_k[l-n+1:l]

    #x_k = X_k*ones(n,1)
    #s_k = S_k*ones(n,1)
    return (((x_k + alpha_aff_prim*delta_x_aff)'*(s_k + alpha_aff_dual*delta_s_aff))/n)[1]

end

function get_sigma(mu,mu_aff)
    return (mu_aff/mu)^3
end

function get_corrector_newton_step_rhs(rc_k,rb_k,X_k,S_k,sigma,mu,predictor_p_k)
    n = size(X_k,1)
    l = size(predictor_p_k,1)
    delta_x_aff = predictor_p_k[1:n,:]
    delta_s_aff = predictor_p_k[l-n+1:l,:]
    delta_X_aff = diagm(vec(delta_x_aff))
    delta_S_aff = diagm(vec(delta_s_aff))

    return [-rc_k; -rb_k; (-1*X_k*S_k*ones(n,1) - delta_X_aff*delta_S_aff*ones(n,1) + sigma*mu*ones(n,1)) ]
end

function get_alpha_max(x_k,s_k,corrector_p_k)
    n = size(x_k,1)
    l = length(corrector_p_k)
    delta_x = corrector_p_k[1:n,:]
    delta_s = corrector_p_k[l-n+1:l,:]
    ## TODO:  Protect againt division by zero
    alpha_max_prim = get_min_for_negative_delta(x_k,delta_x)
    alpha_max_dual = get_min_for_negative_delta(s_k,delta_s)

    return alpha_max_prim,alpha_max_dual
end

function get_alpha(alpha_max_prim,alpha_max_dual)
    return min(1,eta*alpha_max_prim),min(1,eta*alpha_max_dual)
end

function take_step(x_k,s_k,lambda_k,alpha_primal_k,alpha_dual_k,corrector_p_k)
    n = size(x_k,1)
    l = size(corrector_p_k,1)
    delta_x = corrector_p_k[1:n,:]
    delta_s = corrector_p_k[l-n+1:l,:]
    delta_lambda = corrector_p_k[n+1:l-n,:]

    x_k = x_k + alpha_primal_k*delta_x
    s_k = s_k + alpha_dual_k*delta_s
    lambda_k = lambda_k + alpha_dual_k*delta_lambda

    return x_k,s_k,lambda_k
end

function get_D2(X_k, S_k)
  return inv(S_k)*X_k
end
function predictor_corrector(A, c, b, x_0, s_0, lambda_0)
    # initialize variables
    k = 0
    x_k = copy(x_0)
    s_k = copy(s_0)
    lambda_k = copy(lambda_0)


    while k <= max_iter
        X_k = get_X(x_k)
        S_k = get_S(s_k)
        D2 = get_D2(X_k,S_k)
        J = get_jacobian(X_k,S_k, A)
        rc_k = get_rc(A,lambda_k,s_k,c)
        rb_k = get_rb(A,x_k,b)

        predictor_right = get_predcitor_newton_step_rhs(rc_k,rb_k,X_k,S_k)

        # TODO: check for singularity of J
        predictor_p_k = J\predictor_right

        # step length for affine step
        alpha_aff_prim,alpha_aff_dual = get_alpha_affine(x_k,s_k,predictor_p_k)

        # Current Duality measure
        mu = get_mu(x_k,s_k)
        # Duality measure for affine step
        mu_aff = get_mu_aff(alpha_aff_prim,alpha_aff_dual,x_k,s_k,predictor_p_k)

        # Set the Centering Parameter
        sigma = get_sigma(mu,mu_aff)

        # CORRECTOR STEP

        corrector_right = get_corrector_newton_step_rhs(rc_k,rb_k,X_k,S_k,sigma,mu,predictor_p_k)

        corrector_p_k = J\corrector_right

        # max step length
        alpha_max_prim,alpha_max_dual = get_alpha_max(x_k,s_k,corrector_p_k)

        # Final Step length for primal and dual variables

        alpha_primal_k, alpha_dual_k = get_alpha(alpha_max_prim,alpha_max_dual)

        # Take a step in the final search direction
        x_k,s_k,lambda_k = take_step(x_k,s_k,lambda_k,alpha_primal_k,alpha_dual_k,corrector_p_k)

        println("Iteration: ", k)
        @show(x_k)
        @show(s_k)
        #@show(x_k's_k)

        k += 1
    end
end

function get_starting_point(A, b, c)

  #x_hat, lambda_hat, s_hat arre solutions of :
  # min 0.5*x'x subject to Ax=b
  # min 0.5*s's subject to A'lambda + s = c
  x_hat = A'*inv(A*A')*b
  lambda_hat = inv(A*A')*A*c
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

function main()
  A1 = [-2.0 1;
        -1 2;
        1 0]
  b = [2.0; 7; 3]

  AS = [A1 eye(3)]   #problem with slack
  cs = [-1 -2 0 0 0]'
  m,n = size(AS)

  println("Random Initialization:")
  x_0 = ones(n,1)
  s_0 = ones(n,1)
  lambda_0 = ones(m,1)
  predictor_corrector(AS, cs, b, x_0, s_0, lambda_0)

  println("Optimized Initialization")
  x_0, lambda_0, s_0 = get_starting_point(AS, b, cs)
  predictor_corrector(AS, cs, b, x_0, s_0, lambda_0)
end

# This is the public interface for the problem
function iplp(Problem, tol; maxit=100)


end