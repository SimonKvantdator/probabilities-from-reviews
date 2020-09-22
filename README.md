### How do you quantitatively compare two options with a different number of reviews?

If you want to, say, find a nice pizzeria, and there are two pizzerias in your area, one of which has two 5-star reviews, and the other one has four 5-star reviews and three 4-star reviews, how do you choose? This python script takes Bayesian approach to comparing the two options. It uses a Markov Chain Monte Carlo simulation to approximate the probability distribution of the expectation value of a new review of each option.

The model consists of set of probabilities p1, p2, p3, p4, p5 (with p5 = 1 - p1 - p2 - p3 - p4) of a review being 1, 2, 3, 4, or 5 stars respectively. It uses a flat prior, i.e. prior(p1, p2, p3, p4) = (0 <= p1) * (0 <= p2) * (0 <= p3) * (0 <= p4) (p1 + p2 + p3 + p4 <= 1).

There is also an attempt at an analytic solution using Mathematica. The conclusion seems to be that it's difficult.

### How do you use this code?

Put the number of each review score in **reviews.json**. For example, if your local pizzeria only has two 5-star reviews, the json should look like this:
```
{
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 2
}
```
Then you can tweak some parameters in **configs.json** if you'd like. Then run **probabilities_from_reviews.py** and it will produce some plots for you.


