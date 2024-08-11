# Machine Learning
## Logistic regression training
<p style="text-align:justify; text-justify:inter-ideograph;">
There is the logistic regression training interface and use "logistic.py" as an example.
</p>
<p style="text-align:justify; text-justify:inter-ideograph;">
step1:input data
</p>
<pre><code>training_x = secret_fix.Tensor([100000, 32, 32])
training_y = secret_int.Tensor([100000, 10])

test_x = sfix.Tensor([10000, 32, 32])
test_y = sint.Tensor([10000, 10])
</code></pre>

<p style="text-align:justify; text-justify:inter-ideograph;">
step2:model training
</p>
<pre><code>logistic = component.Logistic(32, 2, program)
logistic.fit(training_x, training_y)
</code></pre>
<p style="text-align:justify; text-justify:inter-ideograph;">
You can use predict() to predict labels and calculate predict probabilities. The following outputs the correctness and a measure of how much  the probability estimate is:
</p>
<pre><code>print('%s',(logistic.predict(test_x) - test_y.get_vector()).reveal())
print('%s',(logistic.predict_proba(test_x) - test_y.get_vector()).reveal())
</code></pre>


<p style="text-align:justify; text-justify:inter-ideograph;">
step3:model export
</p>
<pre><code>logistic.reveal_model_to_client()
</code></pre>