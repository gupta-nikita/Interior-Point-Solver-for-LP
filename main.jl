# Parameters
max_iter = 500
eta = 0.9


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

function get_alpha_affine(X_k,S_k,predictor_p_k)
    n = size(X_k,1)
    l = size(predictor_p_k,1)
    delta_x_aff = predictor_p_k[1:n,:]
    delta_s_aff = predictor_p_k[l-n+1:l,:]
    ## TODO:  Protect againt division by zero
    alpha_aff_prim = min(1,minimum(X_k*ones(n,1)./delta_x_aff))
    alpha_aff_dual = min(1,minimum(S_k*ones(n,1)./delta_s_aff))

    return alpha_aff_prim,alpha_aff_dual
end

function get_mu(X_k,S_k)
    n = size(X_k,1)
    return ((X_k*ones(n,1))'*(S_k*ones(n,1))/n)[1]
end


function get_mu_aff(alpha_aff_prim,alpha_aff_dual,X_k,S_k,predictor_p_k)
    n = size(X_k,1)
    l = length(predictor_p_k)
    delta_x_aff = predictor_p_k[1:n]
    delta_s_aff = predictor_p_k[l-n+1:l]

    x_k = X_k*ones(n,1)
    s_k = S_k*ones(n,1)
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

function get_alpha_max(X_k,S_k,corrector_p_k)
    n = size(X_k,1)
    l = length(corrector_p_k)
    delta_x = corrector_p_k[1:n,:]
    delta_s = corrector_p_k[l-n+1:l,:]
    ## TODO:  Protect againt division by zero
    alpha_max_prim = min(1,minimum(X_k*ones(n,1)./delta_x))
    alpha_max_dual = min(1,minimum(S_k*ones(n,1)./delta_s))

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
    s_k = x_k + alpha_dual_k*delta_s
    lambda_k = lambda_k + alpha_dual_k*delta_lambda

    return x_k,s_k,lambda_k
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
        J = get_jacobian(X_k,S_k, A)
        rc_k = get_rc(A,lambda_k,s_k,c)
        rb_k = get_rb(A,x_k,b)

        predictor_right = get_predcitor_newton_step_rhs(rc_k,rb_k,X_k,S_k)

        # TODO: check for singularity of J
        predictor_p_k = J\predictor_right

        # step length for affine step
        alpha_aff_prim,alpha_aff_dual = get_alpha_affine(X_k,S_k,predictor_p_k)

        # Current Duality measure
        mu = get_mu(X_k,S_k)
        # Duality measure for affine step
        mu_aff = get_mu_aff(alpha_aff_prim,alpha_aff_dual,X_k,S_k,predictor_p_k) 
        
        # Set the Centering Parameter
        sigma = get_sigma(mu,mu_aff)

        # CORRECTOR STEP

        corrector_right = get_corrector_newton_step_rhs(rc_k,rb_k,X_k,S_k,sigma,mu,predictor_p_k)

        corrector_p_k = J\corrector_right

        # max step length
        alpha_max_prim,alpha_max_dual = get_alpha_max(X_k,S_k,corrector_p_k)

        # Final Step length for primal and dual variables

        alpha_primal_k, alpha_dual_k = get_alpha(alpha_max_prim,alpha_max_dual)

        # Take a step in the final search direction
        x_k,s_k,lambda_k = take_step(x_k,s_k,lambda_k,alpha_primal_k,alpha_dual_k,corrector_p_k)

        println("Iteration: ", k)
        println("x_k")
        @show(x_k)
        println("s_k")
        @show(s_k)
        println("product")
        @show(x_k's_k)

        k += 1

    end
end