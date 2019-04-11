$(document).ready(function(){
    console.log('Doc is ready')

    $('#evaluate').click(async function() {
        console.log('button was clicked')

        const subtotal = parseFloat($('#subtotal').val());
        const ship = parseFloat($('#ship').val());
        const avg_price_item = parseFloat($('#avg_price_item').val());
        const subscriber_newsletter = parseFloat($('#subscriber_newsletter').val());
        const uses_coupons = parseFloat($('#uses_coupons').val());
        const customer_group = $('#customer_group').val();
        const affiliation = $('#affiliation').val();

        const data = {
            subtotal,
            ship,
            avg_price_item, 
            subscriber_newsletter,
            uses_coupons,
            customer_group,
            affiliation
        }
        console.log(data)

        const response = await $.ajax('/inference', {

            data: JSON.stringify(data),
            method: "post",
            contentType: "application/json"
        })
        console.log(response)
        $('#predictor').val(response.prediction)
    })
})