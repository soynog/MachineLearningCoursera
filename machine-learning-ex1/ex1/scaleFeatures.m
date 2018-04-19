% pass it a matrix of features X, and mu (means) and sigma (stds) vectors, and
% returns a matrix of scaled features
function scaled_feats = scaleFeatures(X, mu, sigma)

scaled_feats = (X - mu) ./ sigma;
