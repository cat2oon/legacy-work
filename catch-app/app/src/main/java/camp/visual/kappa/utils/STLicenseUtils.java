package camp.visual.kappa.utils;

import android.content.Context;
import android.content.SharedPreferences;

import com.sensetime.stmobile.STMobileAuthentificationNative;


public class STLicenseUtils {
    private final static String PREF_ACTIVATE_CODE = "activate_code";
    private final static String PREF_ACTIVATE_CODE_FILE = "activate_code_file";

    public static boolean checkLicense(Context context) {
        String license = "############################################################\n" +
            "# SenseTime License\n" +
            "# License Product: SenseME\n" +
            "# Expiration: 20181217~20190117\n" +
            "# License SN: 42703d09-fe60-47bf-940c-80492d0240a5\n" +
            "############################################################\n" +
            "sGfdd28aT2wZMvt/rIBjvmYKxJ1ufxJMysu3yYMLiRbnSJGHHVJ+N1MQyShM\n" +
            "ttFH3qUnY6yL/cPX3yOYW29S3yqnGCXk3Hff9yDSRbS6arII8qbNqQFFzQtE\n" +
            "LDIvJE5AjhdUilB0mau0h+mj6htGCihrHt8eBWl4SCiDZIB78QbgFFLUAQAA\n" +
            "AAEAAABJgqBOMAsJVVFbvnbBhxpRTHEw5zVhOnmN3YqzPNoh6ps4CZPnu0NH\n" +
            "CIIRuRPvavVh/HeZZnsRj9FBKzvxxnFSHrlWsj4ZFqxilDmTJSn6jHnaYgG+\n" +
            "yazRHPgb2FmugPblmVVoDtjgIZP9i4mZjJoSMHIJl1dficYmpBxlrMboGqIG\n" +
            "EiP8mrCf8fqrGHOUBAbMnwjkBjY1C9ruzhDRvyVk6n380UfUdyABzb7GPZtt\n" +
            "IlmuvaiAKLKNfaTdpGAtj1TFdoXLdF39F4Kua09YT1iRUjm1qUiJJsKH317A\n" +
            "Q5F/eUuCvt3AASTpYPb1GfwgeGv2dhE6kmHzHl/6VQvd6W65AQABAAAAAAAD\n" +
            "AAAAAAAAAAAAAABYMlM2TbplQ+Fy6KZ8tsvvljf3xPq5kUk+BDg8DIFPOLUB\n" +
            "14uhEBES/OXX76c2wx1jU8BvxGZhQSTbj6JeXeEVsVgfQrpKoqWQPRngT+3t\n" +
            "zKnvxCJScSr6UkSkeAjHodTy9TnUX11Ev5Nsk4oBqR7vAGJQHkHCzsfRJyCe\n" +
            "YibUu4S1QqPozlQjIf99oEFOFafXencc1is/QW8aQVgKIS8YbGCiIi7LPAeg\n" +
            "PkicJyG6lXDhlx3TQZmbFfDyDP5pWICTv/XG2J16rLDPhaXdcS8iNNlt0yUI\n" +
            "/m2RtP7Mq0HWDeY6qgctwm2J3CrfP5JROPg+5dIaEHSXqW4s+K68y9dvwsaN\n" +
            "Ft4UrIwNvUuSu2DNKb2hIn/22LkWvnCDio97n+fDC3LWVK9VuLUfLIin/ID1\n" +
            "pKaKs8nI3gRmIYJHqx3hfnkwzf+74rp0KHCRj7rYEakG/J27SxouS9qmnmMv\n" +
            "aXdpC1YUJsA=\n" +
            "------------------------------------------------------------\n" +
            "sGfddx69TAejjkrO0YOw2gzKj9olMTo05scMN3Kvmd7CNja7Re649aFA+cAw\n" +
            "1b78MJiyzq68gGIPoe805rnwLlx8hoOg4jLrLGN9AynupktpFmvCQ7YPYvNX\n" +
            "vKb3dBngCRegewcagWZ/xcoDwW49ewWwkltiUa+6+gibPIaf0VhP5ISbAQAA\n" +
            "AAIAAABug6erX7u9O6+2q8w0E8CM34YdVsWepvLNYbgQwIBykO9fnAZ0kCu/\n" +
            "AzIsAyqnpLDYmUUIYR+M2l2rJIuduLRF+Strh+Ik68lHYYR50ijJnJPfOxoC\n" +
            "NrHYVXS63Bs9jdf+JsJWWbsu99Uq1BdLCECymaUC46d5bgftRup4HOZGQad7\n" +
            "XXK3bUa6Ex/BiWb0Vg+yNMWxFWhWXXp+4QfWsOA5pfC3ij4mQDIu/iQsS3ZZ\n" +
            "VLxMg7NOAOKxrxXAoh1wtNsMe70h8cUtA3p2zIXege3Th8iI8PHMViA7ewOo\n" +
            "Dr+lSjcssDvfeXzPVu6IQVv3sL40Sg1mAJ1rFfJho0EHZEYpAAAAAAAAAAAA\n" +
            "AAAAAAAAAAAAAADiuhPaocwu4TNeqpeh2zWJxzZvg3/sdPm5Hxabh9l9EEE2\n" +
            "QNgBhg3COvzL5/Zg7tYR0a0EmYYAmgVD4vN6r8viO5Pqu2dXIQQWkiq2oHqC\n" +
            "2211GYQ4se2TIxfwbRo8VVdHum/AA8G3ZLnoq4tnU54kGqbfmWf7n9Zw1Rxa\n" +
            "20u7tV3LcRsQ5/IoSmfp3KfRjJegN594OXcgWMMUAzdr2FjRTBtSM3fGQ4Yk\n" +
            "XmpcTwFQPPkTwzTB1wyR2Q3gUygJL47tJQWLEoyTZ0EqgSCpUdq4R4LIr2IF\n" +
            "lUBEvhptH8lNOx4tXyNcXNswxVeonm8kZIzAI8kXi4GIg58LwHoDHsrn5tJz\n" +
            "ykHh2xeGv29cBLO23jhhYibpsrq015wNmfx55xckxMmxTSv3EIIjVsgVp0JJ\n" +
            "p/RKhZvOM7RWXttDp9L+4Am8NlqXoMnI8WH5g+P51wmGRszOA5YcXyT7JiV0\n" +
            "1Mi6YPwCvIjRH2OMoiN4se7tukenHilO5tmPD7wyJfPSG/QszFt+KjvBg9Ps\n" +
            "wYlLDkVYRB8pEqkDF9mSDZTgRhEzHws73Zfm6hNPP9JGcvpL4wW4VeQ2ACbt\n" +
            "tq4GmDj53hBl7d96rEUW71MMUUQQM/SMkuzeoLh24w3YPN6i2SJ2DitJB7MQ\n" +
            "X7XPjZWhbwFeLH84WAx/bAEaxXxW7+NM49HNPBJav9RY/+MMZEvLwYJ3nM/k\n" +
            "AljQm8Hz44wfR65OMZvcnXquK6J+3j2JluILcOf3HA1ns+hmxN0AQhGdAK2P\n" +
            "tkiyu4ViAN0fDzhQ5bq1V55PGYR/rp625CW+Xyqe9524oK6hilcdqQiZ9be5\n" +
            "9+3ZLMENg5lfgHkfFCL1olFcN2AwRsQfdUz3nX+eoUW13JC8Mh2UYFVKB2Ul\n" +
            "9lGYQcX93KoyhE/SdHFsUEAKtpdf2w0Ea0oAZCxfka40MAUguzqIbqOg5Emx\n" +
            "x0oWjR3T46MZR9HcCFwZEsYKAagstjjXLZWSD1kxgKgqVStEXmC0Xx+SCwpU\n" +
            "I9CSddt7sATERHE95SAMIpz3UePFY6gPY1Aq2LvQbG5jGKXQcxDwIH/Fp7Xc\n" +
            "oTYjt2cbc55jdHg1FHuL1iezbMLFUsaTseiXWc6rJV9FpkujuDBk1qHte+2/\n" +
            "LPRo34dld12t5FNFoiYsCoDo7CJOXyOkIyA300gWbC6x0ewDDxw4Pbb3R9SR\n" +
            "CwGq6v17Tawvz0El38nJt7/ce4O7N83hsKFvEFq4B5WVToljub3O/HUBN59R\n" +
            "T0g1VEBGZSH5qb8R2JRmO1lvItoNTF0fvn50thp48ls49D2iqmO4MFBAww3/\n" +
            "qpmdNJtPFhfSvJeypgLTdZQsQgRz8Tc/JZcrF2AAPKRkal4AC5k9/D5AMXGg\n" +
            "WS86EPYJI5nRgV5vj+iHLmd1zrUF7mq43Wnzy3GIvkALVB+hxweUKF1POxy1\n" +
            "pMwqtjKuoIxW76bY0zoXr6zsJVhZgjHCrfF8y89WlUchgcAc46NQ9+1Xwwb0\n" +
            "z0LdNtr1bsitxFTvPktDsErseNJLSrV2c7klXb/I4HEcp5d2gqOd4ztvZigb\n" +
            "YnKDBo29HLJXWEDjPElcdjdpBX+2kdL2AXZAloiypRK7htT2cmPPLm2kfGov\n" +
            "nWxyfHsfYbIuoP3CcM5zFsxkNzM3H1WNn9cKqzvhx4PupZq8BMgR052d59dq\n" +
            "3u1Sf4cU3mfJklsqOgrmeoFACbbMpr1Eg1cujMmuDXjx/LUs6qBg0Le2MFXZ\n" +
            "g5hvVAKsDRXCCiO+WIkqn2Tgpd2TKtIIWqfv0JA7fVReEQ5UN5dYCfg+gDo9\n" +
            "KGSYvRNPa4tN8LSkirGBKMVZQDEmbzdC27Oh95YJvcFHoA6jStYsjx1Gc1xP\n" +
            "CLHkiQbzc0A6jiPmBdOTEiY8w21005vGOWjdj+e3h71xsizyGGuP7FE3Tofo\n" +
            "rOw2LisOVUd2FIY4lYdgUWp8zGsTLQXTYdIfyovgWQnV2l+r8huGxql/0MZx\n" +
            "ZWN7oveHC+j+2muHT4ErU46iHsB63LyegyxmSik=\n" +
            "############################################################";


        String activateCode = "4rol8YX4ba9wXLLmv5Mcw5dLZJhqq3by7VxCwJ+ydxUAPQbIHZVPm26ojKugPavWHcn9VMTSRMBBUOjgC7el";       // S8
        // String activateCode = "4rol8YX4ba9wXLL7l8YuiYJOcpFmqCSvsF4RkMa2LRcCPlLIHZVPm26ojKugPavWHcn9VMTSRMBBUOjgC7el\n";  // Nexus
        // String activateCode = "4rol8YX4ba9wXLLmv5Mcw5dLZMU3rC/3t1sVl5+3IkUAaVPIHZVPm26ojKugPavWHcn9VMTSRMBBUOjgC7el";    // CH S8

        if (isWrongActivateCode(context, license, activateCode)) {
            activateCode = generateActivateCode(context, license);

            if (activateCode == null || activateCode.length() <= 0) {
                return false;
            }

            SharedPreferences sp = context.getApplicationContext()
                .getSharedPreferences(PREF_ACTIVATE_CODE_FILE, Context.MODE_PRIVATE);
            SharedPreferences.Editor editor = sp.edit();
            editor.putString(PREF_ACTIVATE_CODE, activateCode);
            editor.apply();
        }

        return true;
    }

    private static String generateActivateCode(Context ctx, String license) {
        return STMobileAuthentificationNative.generateActiveCodeFromBuffer(ctx, license, license.length());
    }

    private static boolean isWrongActivateCode(Context ctx, String license, String activateCode) {
        if (activateCode == null) {
            return false;
        }

        return STMobileAuthentificationNative.checkActiveCodeFromBuffer(ctx, license, license.length(),
            activateCode, activateCode.length()) != 0;
    }

}
